
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
        """ Configure optimizer for PyTorch Lightning. """
        raise NotImplementedError("Subclasses must implement configure_optimizers()")
    
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

        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.learning_rate, betas=(0.5, 0.999))
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=self.learning_rate, betas=(0.5, 0.999))
        return [opt_d, opt_g], [] # standard Lightning format: (optimizers, schedulers)

    def training_step(self, batch, batch_idx):
        opt_d, opt_g = self.optimizers()
        prev_bars, curr_bars, chord_idx = batch
        batch_size = prev_bars.size(0)

        # Tools & Targets
        real_label = torch.full((batch_size, 1), 0.9, device=self.device)
        fake_label = torch.zeros((batch_size, 1), device=self.device)
        valid_label = torch.ones((batch_size, 1), device=self.device)

        # =========================================================================
        # 1. TRAIN DISCRIMINATOR
        # =========================================================================
        # Generate fake batch (gradients not needed for G here)
        z = torch.randn(batch_size, self.noise_dim, device=self.device)
        fake_bars, _ = self.generator(z, prev_bars, chord_idx)
        
        # Forward Pass: Real
        real_pred, real_feats = self.discriminator(curr_bars, prev_bars, chord_idx)
        d_loss_real = self.criterion(real_pred, real_label) # real_label = 0.9

        # Forward Pass: Fake
        fake_pred_det, _ = self.discriminator(fake_bars.detach(), prev_bars, chord_idx)
        d_loss_fake = self.criterion(fake_pred_det, fake_label) # fake_label = 0

        # Update D
        d_loss = (d_loss_real + d_loss_fake) / 2
        opt_d.zero_grad()
        self.manual_backward(d_loss)
        opt_d.step()

        # Metrics: Accuracy
        acc_real = (real_pred > 0.5).float().mean()
        acc_fake = (fake_pred_det < 0.5).float().mean()
        d_acc = (acc_real + acc_fake) / 2

        # =========================================================================
        # 2. TRAIN GENERATOR
        # =========================================================================
        # Reuse 'fake_bars' from above (preserving gradients for G)
        
        # Forward Pass D (on fake, keeping gradients for G)
        fake_pred, fake_feats = self.discriminator(fake_bars, prev_bars, chord_idx)
        
        # A. Adversarial Loss (G tries to fool D)
        g_loss_adv = self.criterion(fake_pred, valid_label) # valid_label = 1

        # B. Feature Matching Loss (Stability)
        mean_real_f = torch.mean(real_feats.detach(), dim=0)
        mean_fake_f = torch.mean(fake_feats, dim=0)
        g_loss_fm = torch.mean((mean_real_f - mean_fake_f) ** 2)

        # Update G
        g_loss = g_loss_adv + self.feature_matching_weight * g_loss_fm
        
        opt_g.zero_grad()
        self.manual_backward(g_loss)
        opt_g.step()

        # =========================================================================
        # 3. LOGGING
        # =========================================================================
        self.log_dict({
            "d_loss": d_loss,
            "d_accuracy": d_acc,
            "g_loss": g_loss,
            "g_adv": g_loss_adv,
            "g_fm": g_loss_fm
        }, prog_bar=True) 

    def validation_step(self, batch, batch_idx):
        prev_bars, curr_bars, chord_idx = batch
        batch_size = prev_bars.size(0)
        
        # Genera fake
        noise = torch.randn(batch_size, self.noise_dim, device=self.device)
        generated_bars, _ = self.generator(noise, prev_bars, chord_idx)
        
        # Valuta col discriminatore (senza aggiornare gradienti)
        fake_output, _ = self.discriminator(generated_bars, prev_bars, chord_idx)
        
        # Calcola loss (quanto bene il generatore inganna il discriminatore su dati mai visti)
        val_g_loss = self.criterion(fake_output, torch.ones_like(fake_output))
        
        # Logging
        self.log("val_g_loss", val_g_loss, prog_bar=True, synchronization_dist=True)
        
    def test_step(self, batch, batch_idx):
        pass # Implementare se necessario