import torch
import torch.nn as nn
import torch.onnx
import pytorch_lightning as pl

# =================================================================================
# 1. DEFINIZIONE DELLE CLASSI !! codice vecchio, non lo abbiamo usato alla fine
# =================================================================================

class BaseModel(pl.LightningModule):
    """
    Base model class for all neural network models in the project.
    """
    def __init__(self, criterion: nn.Module, learning_rate: float = 1e-3):
        super(BaseModel, self).__init__()
        self.learning_rate = learning_rate
        self.criterion = criterion

    def configure_optimizers(self) -> torch.optim.Optimizer:
        raise NotImplementedError("Subclasses must implement configure_optimizers()")
    
    def get_num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def summary(self):
        print(f"\n{'='*80}")
        print(f"Model: {self.__class__.__name__}")
        print(f"{'='*80}")
        print(self)
        print(f"Total trainable parameters: {self.get_num_parameters():,}")
        print(f"{'='*80}\n")


class Generator(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        self.chord_dim = 12
        
        # Embeddings
        self.chord_embedding = nn.Embedding(num_embeddings=25, embedding_dim=self.chord_dim)
        
        # Noise projection & Memory
        self.project_process = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
        )

        self.gru = nn.GRUCell(input_size=512, hidden_size=512)

        self.reshape_layer = nn.Unflatten(dim=1, unflattened_size=(256, 1, 2)) # 512 -> 256x1x2

        # Conditioning Network (Processes previous bar)
        self.cond_layer1 = self._conv_block(1, 256, (128, 1), 1)
        self.cond_layer2 = self._conv_block(256, 256, (1, 2), (1, 2))
        self.cond_layer3 = self._conv_block(256, 256, (1, 2), (1, 2))
        self.cond_layer4 = self._conv_block(256, 256, (1, 2), (1, 2))

        # Generation Network (Transposed Convolutions)
        # Input channels: 256 (prev stage) + 256 (condition) + chord_dim
        in_ch = 512 + self.chord_dim
        self.gen_layer1 = self._transp_conv_block(in_ch, 256, (1, 2), (1, 2))
        self.gen_layer2 = self._transp_conv_block(in_ch, 256, (1, 2), (1, 2))
        self.gen_layer3 = self._transp_conv_block(in_ch, 256, (1, 2), (1, 2))

        # Final output layer
        self.gen_layer4 = nn.Sequential(
            nn.ConvTranspose2d(in_ch, 1, kernel_size=(128, 1), stride=1),
            nn.Sigmoid()
        )

    def _conv_block(self, in_channels, out_channels, k_size, stride):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, k_size, stride),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def _transp_conv_block(self, in_channels, out_channels, k_size, stride):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, k_size, stride),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def _concat_chords(self, feature_map, cond_map, chord_vec):
        """Concatenates feature map, conditioning map, and expanded chord vector."""
        b, _, h, w = feature_map.shape
        # Expand chords: (B, chord_dim) -> (B, chord_dim, H, W)
        chord_expanded = chord_vec.view(b, self.chord_dim, 1, 1).expand(b, self.chord_dim, h, w)
        return torch.cat((feature_map, cond_map, chord_expanded), dim=1)

    def forward(self, z, condition_matrix, chord_idx, hidden_state=None):
        # 1. Process Conditioning (Previous Bar)
        condition_step1 = self.cond_layer1(condition_matrix)
        condition_step2 = self.cond_layer2(condition_step1)
        condition_step3 = self.cond_layer3(condition_step2)
        condition_step4 = self.cond_layer4(condition_step3)

        # 2. Update Temporal Memory (GRU)
        gru_input = condition_step4.view(z.size(0), -1)
        if hidden_state is None:
            hidden_state = torch.zeros(z.size(0), 512).to(z.device)
        
        new_hidden_state = self.gru(gru_input, hidden_state)

        # 3. Process Noise & Project
        proj = self.project_process(z)
        # Combine with memory and reshape
        base_features = self.reshape_layer(proj + new_hidden_state) 
        
        # 4. Generator Upsampling Steps
        chord_vec = self.chord_embedding(chord_idx)
        
        # Step 1
        merged_step1 = self._concat_chords(base_features, condition_step4, chord_vec)
        gen_step1 = self.gen_layer1(merged_step1)

        # Step 2
        merged_step2 = self._concat_chords(gen_step1, condition_step3, chord_vec)
        gen_step2 = self.gen_layer2(merged_step2)

        # Step 3
        merged_step3 = self._concat_chords(gen_step2, condition_step2, chord_vec)
        gen_step3 = self.gen_layer3(merged_step3)

        # Step 4 (Final)
        merged_step4 = self._concat_chords(gen_step3, condition_step1, chord_vec)
        final_out = self.gen_layer4(merged_step4)
    
        return final_out, new_hidden_state
    

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.chord_dim = 12
        self.chord_embedding = nn.Embedding(num_embeddings=25, embedding_dim=self.chord_dim)

        # Feature Extractor (Convolutional Layers)
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, (128, 2), (1, 2)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, (1, 4), (1, 2)),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Classifier (Fully Connected)
        # Input features: 64 channels * 3 * 2 spatial + chord_dim
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(64*6 + self.chord_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x_curr, x_prev, chord_idx):
        chord_vec = self.chord_embedding(chord_idx)

        # Extract features
        feat_curr = self.conv_layers(x_curr)
        feat_prev = self.conv_layers(x_prev)

        # Flatten & Concat
        flat_curr = self.flatten(feat_curr)
        flat_prev = self.flatten(feat_prev)
        combined = torch.cat((flat_curr, flat_prev, chord_vec), dim=1)
        
        # Classification
        validity = self.classifier(combined)

        return validity, feat_curr
    

class PianoGAN(BaseModel):
    def __init__(self, noise_dim: int = 100, learning_rate: float = 0.0002, feature_matching_weight: float = 1.0):
        super().__init__(criterion=nn.BCELoss(), learning_rate=learning_rate)
        self.save_hyperparameters()
        self.noise_dim = noise_dim
        self.feature_matching_weight = feature_matching_weight
        self.generator = Generator(input_size=noise_dim)
        self.discriminator = Discriminator()
        self.automatic_optimization = False

    def forward(self, z: torch.Tensor, prev_bars: torch.Tensor, chord_idx: torch.Tensor) -> torch.Tensor:
        return self.generator(z, prev_bars, chord_idx)
    
    # ... (Il resto dei metodi non serve per l'export ONNX, ma la classe è definita) ...

# =================================================================================
# 2. LOGICA DI ESPORTAZIONE PER NETRON
# =================================================================================

def export_midinet_complete():
    print("🚀 Avvio procedura di esportazione per Netron...")
    
    # Parametri dimensionali (Basati su MidiNet standard e il tuo codice)
    # Z_dim = 100 (default in PianoGAN)
    # Prev_Bar = [Batch, 1, 128, 16] (128 note, 16 step temporali)
    # Chord = Indice tra 0 e 24
    
    z_dim = 100
    batch_size = 1
    
    # 1. Istanziamo il GENERATORE (è la parte interessante da visualizzare)
    model = Generator(input_size=z_dim)
    model.eval()
    
    # 2. Creiamo i Dummy Inputs (Dati finti della forma corretta)
    dummy_z = torch.randn(batch_size, z_dim)
    dummy_prev_bars = torch.randn(batch_size, 1, 128, 16) # [B, C, H, W]
    dummy_chord_idx = torch.tensor([5]) # Un accordo a caso (es. indice 5)
    dummy_hidden = torch.zeros(batch_size, 512) # Stato iniziale GRU

    output_path = "midinet_complete_structure.onnx"

    print(f"📦 Esportazione in corso in: {output_path}")
    print("   - Includerà: Embedding Accordi, GRU, Convoluzioni, Concatenazioni")

    try:
        torch.onnx.export(
            model,
            # La tupla degli input deve seguire l'ordine di Generator.forward
            # def forward(self, z, condition_matrix, chord_idx, hidden_state=None):
            (dummy_z, dummy_prev_bars, dummy_chord_idx, dummy_hidden),
            output_path,
            export_params=True,        # Salva i pesi dentro il file
            opset_version=11,          # Versione stabile
            do_constant_folding=True,
            input_names=['Input_Noise_Z', 'Input_Prev_Bar', 'Input_Chord_ID', 'Input_GRU_Hidden'],
            output_names=['Output_PianoRoll', 'Output_Next_Hidden'],
            dynamic_axes={
                'Input_Noise_Z': {0: 'batch_size'},
                'Input_Prev_Bar': {0: 'batch_size'},
                'Input_Chord_ID': {0: 'batch_size'},
                'Output_PianoRoll': {0: 'batch_size'}
            }
        )
        print(f"✅ FILE CREATO CON SUCCESSO: {output_path}")
        print("👉 Ora vai su https://netron.app e apri questo file.")
        print("   Vedrai chiaramente i blocchi 'Concat' dove l'embedding dell'accordo entra nella rete.")
        
    except Exception as e:
        print(f"❌ Errore critico durante l'export: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    export_midinet_complete()