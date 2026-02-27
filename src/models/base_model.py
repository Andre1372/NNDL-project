
# For model building
import torch
import torch.nn as nn
import pytorch_lightning as pl


class MinibatchStdDev(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # x: (B, C, H, W)
        batch_size, channels, h, w = x.shape
        # Calculate std across batch for each feature
        # std shape: (1, C, H, W)
        std = torch.std(x, dim=0, keepdim=True)
        # Average std over all features
        avg_std = torch.mean(std) # Scalar
        # Expand back to (B, 1, H, W)
        expanded_std = avg_std.expand(batch_size, 1, h, w)
        # Concatenate on channel dim
        return torch.cat([x, expanded_std], dim=1)


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
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
        )

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
        self.res1 = self._residual_block(256)
        self.gen_layer2 = self._transp_conv_block(in_ch, 256, (1, 2), (1, 2))
        self.res2 = self._residual_block(256)
        self.gen_layer3 = self._transp_conv_block(in_ch, 256, (1, 2), (1, 2))
        self.res3 = self._residual_block(256)

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

    def _residual_block(self, channels):
        return nn.Sequential(
            nn.ZeroPad2d((0, 1, 0, 0)), # Padding esplicito per kernel (1, 2): (Left=0, Right=1, Top=0, Bottom=0)
            nn.Conv2d(channels, channels, kernel_size=(1,2), stride=1, padding=0),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.2),
            nn.ZeroPad2d((0, 1, 0, 0)), # Padding esplicito
            nn.Conv2d(channels, channels, kernel_size=(1,2), stride=1, padding=0),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.2),
        )

    def _concat_chords(self, feature_map, cond_map, chord_vec):
        """Concatenates feature map, conditioning map, and expanded chord vector."""
        b, _, h, w = feature_map.shape
        # Expand chords: (B, chord_dim) -> (B, chord_dim, H, W)
        chord_expanded = chord_vec.view(b, self.chord_dim, 1, 1).expand(b, self.chord_dim, h, w)
        return torch.cat((feature_map, cond_map, chord_expanded), dim=1)

    def forward(self, z, condition_matrix, chord_idx):
        # 1. Process Conditioning (Previous Bar)
        condition_step1 = self.cond_layer1(condition_matrix)
        condition_step2 = self.cond_layer2(condition_step1)
        condition_step3 = self.cond_layer3(condition_step2)
        condition_step4 = self.cond_layer4(condition_step3)

        # 2. Process Noise & Project
        proj = self.project_process(z)
        # Reshape to feature map
        base_features = self.reshape_layer(proj)
        
        # 4. Generator Upsampling Steps
        chord_vec = self.chord_embedding(chord_idx)
        
        # Step 1
        merged_step1 = self._concat_chords(base_features, condition_step4, chord_vec)
        gen_step1 = self.gen_layer1(merged_step1)
        gen_step1 = torch.nn.functional.leaky_relu(gen_step1 + self.res1(gen_step1), 0.2)

        # Step 2
        merged_step2 = self._concat_chords(gen_step1, condition_step3, chord_vec)
        gen_step2 = self.gen_layer2(merged_step2)
        gen_step2 = torch.nn.functional.leaky_relu(gen_step2 + self.res2(gen_step2), 0.2)

        # Step 3
        merged_step3 = self._concat_chords(gen_step2, condition_step2, chord_vec)
        gen_step3 = self.gen_layer3(merged_step3)
        gen_step3 = torch.nn.functional.leaky_relu(gen_step3 + self.res3(gen_step3), 0.2)

        # Step 4 (Final)
        merged_step4 = self._concat_chords(gen_step3, condition_step1, chord_vec)
        final_out = self.gen_layer4(merged_step4)

        return final_out, None
    

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.chord_dim = 12
        self.chord_embedding = nn.Embedding(num_embeddings=25, embedding_dim=self.chord_dim)

        # Feature Extractor (Progressive Convolutional Layers)
        # Input: (1, 128, 16)
        self.conv_layers = nn.Sequential(
            # Stage 1: (1, 128, 16) -> (32, 64, 8)
            nn.Conv2d(1, 32, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Stage 2: (32, 64, 8) -> (64, 32, 4)
            nn.Conv2d(32, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Stage 3: (64, 32, 4) -> (128, 16, 2)
            nn.Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.LeakyReLU(0.2, inplace=True),

            # Stage 4: (128, 16, 2) -> (256, 1, 1)
            nn.Conv2d(128, 256, kernel_size=(16, 2), stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Minibatch Standard Deviation
        self.minibatch_std = MinibatchStdDev()
        
        # Classifier (Fully Connected)
        # Input features: (256 channels + 1 std_ch) * 1 * 1 spatial + chord_dim
        # 257 + 12 = 269
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(257 + self.chord_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, x, chord_idx):
        chord_vec = self.chord_embedding(chord_idx)

        # 1. Extract features with progressive convolutions
        feat = self.conv_layers(x)

        # 2. Add Minibatch StdDev for batch diversity analysis
        feat_with_std = self.minibatch_std(feat)

        # 3. Flatten and combine with chords
        out_flat = self.flatten(feat_with_std)
        combined = torch.cat((out_flat, chord_vec), dim=1)
        
        # 4. Classification
        validity = self.classifier(combined)

        return validity, feat
    

class PianoGAN(pl.LightningModule):
    def __init__(self, noise_dim: int = 100, feature_matching_weight: float = 0.0, gradient_penalty_lambda: float = 10.0, n_critic=5):
        """ 
        Initialize the base model. 
        Args:
            noise_dim: Dimension of the input noise vector for the generator.
            feature_matching_weight: Weight for the feature matching loss.
            gradient_penalty_lambda: Weight for the gradient penalty (WGAN-GP).
            n_critic: Number of critic steps per generator step.
        """
        super().__init__()
        
        self.save_hyperparameters()
        self.noise_dim = noise_dim
        self.feature_matching_weight = feature_matching_weight
        self.gradient_penalty_lambda = gradient_penalty_lambda
        self.n_critic = n_critic
        
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
        """ Define the two separate optimizers for Discriminator and Generator.
            Learning rate can be different, ultimately we just set it to the same value for both networks """
        # Generator Learning Rate
        lr_g = 0.0001
        # Critic's Learning Rate
        lr_d = 0.0001

        opt_g = torch.optim.RMSprop(self.generator.parameters(), lr=lr_g)
        opt_d = torch.optim.RMSprop(self.discriminator.parameters(), lr=lr_d)
        
        return [opt_d, opt_g], [] # standard Lightning format: (optimizers, schedulers)

    def compute_gradient_penalty(self, real_samples, fake_samples, chord_idx):
        """Calculates the gradient penalty loss for WGAN GP"""
        # Random weight term for interpolation between real and fake samples
        alpha = torch.rand((real_samples.size(0), 1, 1, 1), device=self.device)
        
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        
        d_interpolates, _ = self.discriminator(interpolates, chord_idx)
        
        fake = torch.ones((real_samples.size(0), 1), device=self.device)
        
        # Get gradient w.r.t. interpolates
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,   # Acts as initial gradient (all 1s) to handle vector output; computes sum of gradients
            create_graph=True,   # Critical: builds graph of the gradient calculation to allow 2nd derivative (backprop through gradient)
            retain_graph=True,   # Keeps the graph in memory so we can reuse it for the final loss.backward() pass
            only_inputs=True,    # Optimization: only compute gradients w.r.t 'interpolates', ignoring model parameters here
        )[0]
        
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def training_step(self, batch, batch_idx):
        opt_d, opt_g = self.optimizers()
        prev_bars, curr_bars, chord_idx = batch
        batch_size = prev_bars.size(0)

        # =========================================================================
        # 1. TRAIN CRITIC
        # =========================================================================
        
        # Generazione fake 
        z = torch.randn(batch_size, self.noise_dim, device=self.device)
        fake_bars, _ = self.generator(z, prev_bars, chord_idx)
        
        # Forward Pass: Real & Fake
        real_validity, _ = self.discriminator(curr_bars, chord_idx)
        fake_validity, _ = self.discriminator(fake_bars.detach(), chord_idx)
        
        # Gradient Penalty
        gradient_penalty = self.compute_gradient_penalty(curr_bars.data, fake_bars.data, chord_idx)
        
        # Adversarial loss (Wasserstein)
        # Loss D (da minimizzare): E[fake] - E[real] + lambda * gp
        w_dist = torch.mean(real_validity) - torch.mean(fake_validity)
        d_loss = -w_dist + self.gradient_penalty_lambda * gradient_penalty
        
        opt_d.zero_grad()
        self.manual_backward(d_loss)
        opt_d.step()
        
        # Logging for Critic        
        self.log("d_loss", d_loss, prog_bar=True)
        self.log("w_distance", w_dist)
        
        # =========================================================================
        # 2. TRAIN GENERATOR (Every n_critic steps)
        # =========================================================================
        
        if batch_idx % self.n_critic == 0:
            # train G getting the validity of the fake bars
            fake_validity_g, fake_feats = self.discriminator(fake_bars, chord_idx)
            
            # A. Adversarial Loss
            # g_loss = -E[Critic(fake)]
            g_loss_adv = -torch.mean(fake_validity_g)

            # B. Feature Matching Loss
            with torch.no_grad():
                _, real_feats = self.discriminator(curr_bars, chord_idx)

            mean_real_f = torch.mean(real_feats, dim=0)
            mean_fake_f = torch.mean(fake_feats, dim=0)
            g_loss_fm = torch.mean((mean_real_f - mean_fake_f) ** 2)

            g_loss = g_loss_adv + self.feature_matching_weight * g_loss_fm
            
            opt_g.zero_grad()
            self.manual_backward(g_loss)
            opt_g.step()

            # Logging
            self.log("g_loss", g_loss, prog_bar=True)
            self.log("g_adv", g_loss_adv)
            self.log("g_fm", g_loss_fm)
            self.log("global_step", float(self.global_step))

    def validation_step(self, batch, batch_idx):
        prev_bars, curr_bars, chord_idx = batch
        batch_size = prev_bars.size(0)
        
        # Genera fake
        noise = torch.randn(batch_size, self.noise_dim, device=self.device)
        generated_bars, _ = self.generator(noise, prev_bars, chord_idx)
        
        # critic's evaluation
        fake_validity, fake_feats = self.discriminator(generated_bars, chord_idx)
        real_validity, real_feats = self.discriminator(curr_bars, chord_idx)
        
        # W-Distance: E[real] - E[fake]
        val_w_dist = torch.mean(real_validity) - torch.mean(fake_validity)
        
        # Feature Matching Loss
        mean_real_f = torch.mean(real_feats, dim=0)
        mean_fake_f = torch.mean(fake_feats, dim=0)
        val_g_fm = torch.mean((mean_real_f - mean_fake_f) ** 2)
        
        # Log
        self.log("val_w_dist", val_w_dist, prog_bar=True)
        self.log("val_g_fm", val_g_fm, prog_bar=True)
        
    def test_step(self, batch, batch_idx):
        ''' test su istogrammi di distribuzione delle note
            questa metrica non l'abbiamo usata nel paper
        '''
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