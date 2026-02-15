
# For model building
import torch
import torch.nn as nn
import pytorch_lightning as pl

class BaseModel(pl.LightningModule):
    """
    Base model class for all neural network models in the project.
    
    This provides a common interface for model saving, loading, and summary.
    """
    
    def __init__(self, criterion: nn.Module, learning_rate: float = 1e-3):
        """ Initialize the base model. """
        super(BaseModel, self).__init__()
        self.learning_rate = learning_rate
        self.criterion = criterion

    def configure_optimizers(self) -> torch.optim.Optimizer:

        raise NotImplementedError("Subclasses must implement configure_optimizers() to return their optimizers.")

    def get_num_parameters(self) -> int:
        """ Get the total number of trainable parameters. """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def summary(self):
        """Print model summary."""
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
        base_features = self.reshape_layer(proj) # + new_hidden_state add to use gru
        
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
        final_out = final_out*1.1 # Amplificazione per compensare la saturazione da Sigmoid
    
        final_out = torch.clamp(final_out, 0, 1)
        return final_out, None
    

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
            nn.Linear(64*3 + self.chord_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x, chord_idx):
        chord_vec = self.chord_embedding(chord_idx)

        # Extract features
        feat_curr = self.conv_layers(x)

        out_flat = self.flatten(feat_curr)

        combined = torch.cat((out_flat, chord_vec), dim=1) # Shape: (B, 231 + 12)
        
        # Classification
        validity = self.classifier(combined)

        return validity, feat_curr
    

class PianoGAN(BaseModel):
    def __init__(self, noise_dim: int = 100, learning_rate: float = 0.0002, feature_matching_weight: float = 5.0):
        """ 
        Initialize the base model. 
        Args:
            noise_dim: Dimension of the input noise vector for the generator.
            learning_rate: Learning rate for both generator and discriminator optimizers.
            feature_matching_weight: Weight for the feature matching loss.
        """
        super().__init__(criterion=nn.BCELoss(), learning_rate=learning_rate)
        
        self.save_hyperparameters()
        self.noise_dim = noise_dim
        self.feature_matching_weight = feature_matching_weight
        
        # Sub-modules
        self.generator = Generator(input_size=noise_dim)
        self.discriminator = Discriminator()
        
        # Important: Disable automatic optimization to manage G and D separately
        self.automatic_optimization = False

    def forward(self, z: torch.Tensor, prev_bars: torch.Tensor, chord_idx: torch.Tensor) -> torch.Tensor:
        """
        Generate a new bar.
        Args:
            z: Noise vector
            prev_bars: Conditioning matrix (previous bar)
            chord_idx: Indices of the chords (0-24)
        """
        return self.generator(z, prev_bars, chord_idx)

    def configure_optimizers(self):
        """ Define the two separate optimizers for Discriminator and Generator. """
        # Learning Rate del Generatore (più alto per permettergli di rincorrere)
        lr_g = 0.0004 
        # Learning Rate del Discriminatore (più basso per frenarlo)
        lr_d = 0.00005

        opt_g = torch.optim.Adam(
            self.generator.parameters(), 
            lr=lr_g, 
            betas=(0.5, 0.999) # Beta1 a 0.5 aiuta la stabilità nelle GAN
        )
        
        opt_d = torch.optim.Adam(
            self.discriminator.parameters(), 
            lr=lr_d, 
            betas=(0.5, 0.999)
        )
        return [opt_d, opt_g], [] # standard Lightning format: (optimizers, schedulers)

    def training_step(self, batch, batch_idx):
        opt_d, opt_g = self.optimizers()
        prev_bars, curr_bars, chord_idx = batch
        batch_size = prev_bars.size(0)

        # 1. Definizione Targets e Noise
        real_label = torch.full((batch_size, 1), 0.8, device=self.device)
        fake_label = torch.zeros((batch_size, 1), device=self.device)
        valid_label = torch.ones((batch_size, 1), device=self.device)

        # 2. Generazione Falsa (Sempre necessaria per G)
        z = torch.randn(batch_size, self.noise_dim, device=self.device)
        fake_bars, _ = self.generator(z, prev_bars, chord_idx)

        # Inizializziamo le variabili che potrebbero non essere calcolate in ogni batch
        # per evitare l'errore UnboundLocalError
        d_loss = None
        d_acc = None

        # =========================================================================
        # 1. TRAIN DISCRIMINATOR (Ogni 3 batch)
        # =========================================================================
        if batch_idx % 3 == 0:
            d_noise = 0.1 * torch.randn_like(curr_bars)
            
            # Forward Pass: Real
            real_pred, _ = self.discriminator(curr_bars + d_noise, chord_idx)
            d_loss_real = self.criterion(real_pred, real_label)

            # Forward Pass: Fake (usiamo .detach() per non influenzare G qui)
            fake_pred_det, _ = self.discriminator(fake_bars.detach() + d_noise, chord_idx)
            d_loss_fake = self.criterion(fake_pred_det, fake_label)

            d_loss = (d_loss_real + d_loss_fake) / 2
            opt_d.zero_grad()
            self.manual_backward(d_loss)
            opt_d.step()

            # Accuratezza per il log
            acc_real = (real_pred > 0.5).float().mean()
            acc_fake = (fake_pred_det < 0.5).float().mean()
            d_acc = (acc_real + acc_fake) / 2

        # =========================================================================
        # 2. TRAIN GENERATOR (Sempre)
        # =========================================================================
        
        # PASSAGGIO CRUCIALE: Dobbiamo estrarre real_feats SEMPRE per la FM Loss.
        # Lo facciamo con torch.no_grad() per non sprecare memoria e non allenare D.
        with torch.no_grad():
            _, real_feats = self.discriminator(curr_bars, chord_idx)

        # Forward D su fake per allenare G
        fake_pred, fake_feats = self.discriminator(fake_bars, chord_idx)
        
        # A. Adversarial Loss
        g_loss_adv = self.criterion(fake_pred, valid_label)

        # B. Feature Matching Loss (real_feats ora è disponibile!)
        mean_real_f = torch.mean(real_feats, dim=0) # Rimosso .detach() perché è già in no_grad
        mean_fake_f = torch.mean(fake_feats, dim=0)
        g_loss_fm = torch.mean((mean_real_f - mean_fake_f) ** 2)

        g_loss = g_loss_adv + self.feature_matching_weight * g_loss_fm
        
        opt_g.zero_grad()
        self.manual_backward(g_loss)
        opt_g.step()

        # =========================================================================
        # 3. LOGGING
        # =========================================================================
        metrics = {
            "g_loss": g_loss,
            "g_adv": g_loss_adv,
            "g_fm": g_loss_fm
        }

        if d_loss is not None:
            metrics.update({
                "d_loss": d_loss,
                "d_accuracy": d_acc
            })

        self.log_dict(metrics, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        prev_bars, curr_bars, chord_idx = batch
        batch_size = prev_bars.size(0)
        
        # Genera fake
        noise = torch.randn(batch_size, self.noise_dim, device=self.device)
        generated_bars, _ = self.generator(noise, prev_bars, chord_idx)
        
        # Valuta col discriminatore (senza aggiornare gradienti)
        fake_output, _ = self.discriminator(generated_bars, chord_idx)
        
        # Calcola loss (quanto bene il generatore inganna il discriminatore su dati mai visti)
        val_g_loss = self.criterion(fake_output, torch.ones_like(fake_output))
        
    def test_step(self, batch, batch_idx):
        prev_bars, _, chord_idx = batch
        batch_size = prev_bars.size(0)

        # 1. Generazione delle barre dal rumore
        z = torch.randn(batch_size, self.noise_dim, device=self.device)
        gen_bars, _ = self.generator(z, prev_bars, chord_idx) #

        # 2. Calcolo Istogramma delle barre generate (Pitch Class)
        # Somma energia su Batch, Channel, Time -> (128 note MIDI)
        gen_pitch_activity = gen_bars.sum(dim=(0, 1, 3)) 
        
        gen_hist = torch.zeros(12, device=self.device)
        for m in range(128):
            gen_hist[m % 12] += gen_pitch_activity[m]
        
        # Normalizzazione
        gen_hist = gen_hist / (gen_hist.sum() + 1e-8)

        # 3. Confronto con l'istogramma del dataset (se disponibile)
        if hasattr(self, 'target_histogram'):
            # Calcoliamo l'errore assoluto medio tra le distribuzioni
            dist = torch.abs(gen_hist - self.target_histogram).mean()
            self.log("test/histogram_error", dist)
            
            # Cosine Similarity (1 = identici, 0 = diversi)
            cos_sim = torch.nn.functional.cosine_similarity(gen_hist.unsqueeze(0), 
                                                            self.target_histogram.unsqueeze(0))
            self.log("test/histogram_similarity", cos_sim)

        # Logghiamo i valori dell'istogramma per l'analisi finale
        for i, val in enumerate(gen_hist):
            self.log(f"test_hist/note_{i}", val)

        return gen_bars # Utile per visualizzazioni finali