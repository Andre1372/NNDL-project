
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
        # Nel __init__ del tuo modello
        # 25 perché 0-23 sono accordi, 24 è silenzio. 12 è una dimensione arbitraria (ma sensata).

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


class SimpleMLP(BaseModel):
    """
    Simple Multi-Layer Perceptron example.
    This is a template/example model architecture.
    """
    
    def __init__(
        self, 
        input_dim: int, 
        hidden_dims: list, 
        output_dim: int, 
        criterion: nn.Module,
        dropout: float = 0.5,
        learning_rate: float = 1e-3
    ):

        super(SimpleMLP, self).__init__(criterion, learning_rate)
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Save a serializable hyperparameter dict on the model so that training exports (checkpoints/configs) can include the model constructor arguments
        hparams_safe = {
            'class_name': self.__class__.__name__,
            'input_dim': input_dim,
            'hidden_dims': hidden_dims,
            'output_dim': output_dim,
            'dropout': dropout,
            'learning_rate': learning_rate,
            'criterion': criterion.__class__.__name__,
        }
        self.save_hyperparameters(hparams_safe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return self.network(x)
    
    def training_step(self, batch: tuple, batch_idx) -> torch.Tensor:
        """
        Training step for PyTorch Lightning.
        Args:
            batch: Tuple of (inputs, targets)
        Returns:
            Loss tensor
        """
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, targets)

        # Log metrics
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch: tuple, batch_idx) -> torch.Tensor:
        """
        Validation step for PyTorch Lightning.
        Args:
            batch: Tuple of (inputs, targets)
        Returns:
            Loss tensor
        """
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, targets)

        # Log metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def test_step(self, batch: tuple, batch_idx) -> torch.Tensor:
        """
        Test step for PyTorch Lightning.
        Args:
            batch: Tuple of (inputs, targets)
        Returns:
            Loss tensor
        """
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, targets)
        
        return loss
    
    def configure_optimizers(self) -> torch.optim.Optimizer:

        return torch.optim.Adam(self.network.parameters(), lr=self.learning_rate)


class Generator(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        self.chord_dim = 12  # Dimensione dell'embedding degli accordi
        self.chord_embedding = nn.Embedding(num_embeddings=25, embedding_dim=self.chord_dim)
        # --- Generator part ---
        # Fully connected layers
        self.ff_nets = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=1024),
            nn.BatchNorm1d(num_features=1024),
            nn.LeakyReLU(0.2), # Passaggio a LeakyReLU per catturare sfumature
            nn.Dropout(0.3), #vediamo se funziona, nel caso tolgo
            nn.Linear(in_features=1024, out_features=512),
            nn.BatchNorm1d(num_features=512),
            nn.LeakyReLU(0.2),
        )
        # Aggiungiamo una cella GRU per la memoria del tema
        self.gru = nn.GRUCell(input_size=512, hidden_size=512)
        
        # Reshape layer (512 x 1 -> 256 x 1 x 2)
        self.reshape = nn.Unflatten(dim=1, unflattened_size=(256, 1, 2))
        # Transposed convolutional layers
        # Input Concat Channels: 256 (from reshape) + 256 (from conditioner) + 12 (from chords) = 524
        self.transp_conv_1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=512+12, out_channels=256, kernel_size=(1, 2), stride=(1, 2)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2)
        )
        self.transp_conv_2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=512+12, out_channels=256, kernel_size=(1, 2), stride=(1, 2)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2)
        )
        self.transp_conv_3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=512+12, out_channels=256, kernel_size=(1, 2), stride=(1, 2)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2)
        )
        self.transp_conv_4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=512+12, out_channels=1, kernel_size=(128, 1), stride=1),
            nn.Sigmoid() # Output values between 0 and 1
        )

        # --- Conditioner part ---
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=256, kernel_size=(128, 1), stride=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2)
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1, 2), stride=(1, 2)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2)
        )
        self.conv_3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1, 2), stride=(1, 2)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2)
        )
        self.conv_4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1, 2), stride=(1, 2)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2)
        )

    def forward(self, z, condition_matrix, chord_idx, hidden_state=None):
        chord_vector = self.chord_embedding(chord_idx)
        # Conditioner process condition_matrix
        condition_step1 = self.conv_1(condition_matrix)
        condition_step2 = self.conv_2(condition_step1)
        condition_step3 = self.conv_3(condition_step2)
        condition_step4 = self.conv_4(condition_step3)

        b = z.size(0)
        gru_input = condition_step4.view(b, -1)
        # 2. Aggiornamento del TEMA (Il cuore della melodia)
        if hidden_state is None:
            hidden_state = torch.zeros(b, 512).to(z.device)
        
        new_hidden_state = self.gru(gru_input, hidden_state)
        # Generator process noise
        linear_output = self.ff_nets(z)
        reshaped_output = self.reshape(linear_output)

        theme_projection = new_hidden_state

        linear_output = self.ff_nets(z)

        combined_features = linear_output + theme_projection

        reshaped_output = self.reshape(combined_features)
        # 3. Helper function per concatenare gli accordi
        # Espande il vettore (B, 12) per matchare le dimensioni H, W del layer corrente
        def concat_chords(feature_map, cond_map, chord_vec):
            # feature_map: (B, C1, H, W)
            # cond_map: (B, C2, H, W)
            # chord_vec: (B, 12)
            b, _, h, w = feature_map.shape
            # Reshape e Expand: (B, 12) -> (B, 12, 1, 1) -> (B, 12, H, W)
            chord_expanded = chord_vec.view(b, self.chord_dim, 1, 1).expand(b, self.chord_dim, h, w)
            # Concatena tutto lungo i canali
            return torch.cat((feature_map, cond_map, chord_expanded), dim=1)
        
        # Step 1: Noise + Cond4 + Chords
        merged1 = concat_chords(reshaped_output, condition_step4, chord_vector) # Canali: 256+256+12 = 524
        output_step1 = self.transp_conv_1(merged1)                      # Output: (B, 256, 1, 4)

        # Step 2: Out1 + Cond3 + Chords
        merged2 = concat_chords(output_step1, condition_step3, chord_vector)         # Canali: 524
        output_step2 = self.transp_conv_2(merged2)                      # Output: (B, 256, 1, 8)

        # Step 3: Out2 + Cond2 + Chords
        merged3 = concat_chords(output_step2, condition_step2, chord_vector)         # Canali: 524
        output_step3 = self.transp_conv_3(merged3)                      # Output: (B, 256, 1, 16)

        # Step 4: Out3 + Cond1 + Chords
        merged4 = concat_chords(output_step3, condition_step1, chord_vector)         # Canali: 524
        output_step4 = self.transp_conv_4(merged4)                      # Output: (B, 1, 128, 16)
    


        return output_step4, new_hidden_state
    

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.chord_embedding = nn.Embedding(num_embeddings=25, embedding_dim=12)

        # Convolutional layers
        self.first_conv = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(128, 2), stride=(1, 2)) # Separated by the other for the feature matching technique
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(128,2), stride=(1, 2)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 4), stride=(1, 2)),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # Flatten layer
        self.flatten = nn.Flatten()
        # Feedforward layers
        self.ff_layers = nn.Sequential(
            nn.Linear(in_features=64*3 + 12, out_features=512), # 12 more for the chords
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(in_features=512, out_features=1),
            nn.Sigmoid()
        )

    def forward(self, x_curr, x_prev, chord_idx):
        """
        Args:
            x: Piano roll corrente (B, 1, 128, 16)
            chord_vector: Embedding accordo (B, 12)
        """
        chord_vector = self.chord_embedding(chord_idx)

        feat_curr = self.conv_layers(x_curr)
        feat_prev = self.conv_layers(x_prev)

        int_feat = feat_curr #eventualmente per feature matching

        flat_curr = self.flatten(feat_curr)
        flat_prev = self.flatten(feat_prev)

        combined = torch.cat((flat_curr, flat_prev, chord_vector), dim=1) # Shape: (B, 231 + 12)
        output = self.ff_layers(combined)

        return output, int_feat
    

class PianoGAN(BaseModel):
    def __init__(self, noise_dim: int = 100, learning_rate: float = 0.0002):
        """ 
        Initialize the base model. 
        Args:
            noise_dim: Dimension of the input noise vector for the generator.
            learning_rate: Learning rate for both generator and discriminator optimizers.
        """
        super().__init__(criterion=nn.BCELoss(), learning_rate=learning_rate)
        
        self.save_hyperparameters()
        self.noise_dim = noise_dim
        
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
        # Lookup embedding
        return self.generator(z, prev_bars, chord_idx)

    def configure_optimizers(self):
        """ Define the two separate optimizers for Discriminator and Generator. """

        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.learning_rate, betas=(0.5, 0.999))
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=self.learning_rate, betas=(0.5, 0.999))
        return [opt_d, opt_g], [] # standard Lightning format: (optimizers, schedulers)

    def training_step(self, batch, batch_idx):
        # Retrieve optimizers
        opt_d, opt_g = self.optimizers()
        
        # Unpacking the batch
        prev_bars, curr_bars, chord_idx = batch
        batch_size = prev_bars.size(0)
        
        # Generate noise
        # Note: Lightning automatically handles the device (self.device)
        noise = torch.randn(batch_size, self.noise_dim, device=self.device)

        # ---------------------
        # 1. Train Discriminator
        # ---------------------
        
        # Generazione fake (senza aggiornare gradienti G per ora)
        generated_bars, _ = self.generator(noise, prev_bars, chord_idx)

        # Label Smoothing: usiamo 0.9 invece di 1.0 per i target reali. 
        # Aiuta a stabilizzare l'apprendimento delle dinamiche (velocity).
        real_labels = torch.full((batch_size, 1), 0.9, device=self.device)
        fake_labels = torch.zeros((batch_size, 1), device=self.device)

        # Forward pass D su real
        real_output, _ = self.discriminator(curr_bars, prev_bars, chord_idx)
        
        # Forward pass D su fake (detach per non propagare su G)
        fake_output_detached, _ = self.discriminator(generated_bars.detach(), prev_bars, chord_idx)

        # Calcolo Loss Discriminator (replicando `discriminator_loss` di test.py)
        # test.py usa BCEWithLogitsLoss.
        # real targets = 1, fake targets = 0
        real_loss = self.criterion(real_output, real_labels)
        fake_loss = self.criterion(fake_output_detached, fake_labels)
        d_loss = (real_loss + fake_loss) / 2

        # Step di ottimizzazione Discriminator
        opt_d.zero_grad()
        self.manual_backward(d_loss)
        opt_d.step()

        # ---------------------
        # 2. Train Generator
        # ---------------------
        valid_labels = torch.ones((batch_size, 1), device=self.device)
        # Forward pass D su fake (questa volta serve il gradiente per G)
        fake_output, _ = self.discriminator(generated_bars, prev_bars, chord_idx)
        
        # Calcolo Loss Generator (replicando `generator_loss` di test.py)
        # Il generatore vuole ingannare il discriminatore, quindi target = 1
        g_loss = self.criterion(fake_output, torch.ones_like(fake_output))

        # Step di ottimizzazione Generator
        opt_g.zero_grad()
        self.manual_backward(g_loss)
        opt_g.step()

        # Logging
        self.log_dict({
            "g_loss": g_loss, 
            "d_loss": d_loss,
            "real_loss": real_loss,
            "fake_loss": fake_loss
        }, prog_bar=True)

        # Non ritorniamo loss tensor perché siamo in manual_optimization
        return 

    # Implementazione dei metodi astratti richiesti da BaseModel
    # che non sono strettamente usati nel training loop di test.py
    # ma devono esistere.
    def validation_step(self, batch, batch_idx):
        pass # Implementare se necessario
        
    def test_step(self, batch, batch_idx):
        pass # Implementare se necessario