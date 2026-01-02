
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

    def forward(self) -> torch.Tensor:
        """ Forward pass through the model."""
        raise NotImplementedError("Subclasses must implement forward()")
    
    def training_step(self, batch: tuple, batch_idx) -> torch.Tensor:
        """ Training step for PyTorch Lightning. """
        raise NotImplementedError("Subclasses must implement training_step()")
    
    def validation_step(self, batch: tuple, batch_idx) -> torch.Tensor:
        """ Validation step for PyTorch Lightning. """
        raise NotImplementedError("Subclasses must implement validation_step()")
        
    def test_step(self, batch: tuple, batch_idx) -> torch.Tensor:
        """ Test step for PyTorch Lightning. """
        raise NotImplementedError("Subclasses must implement test_step()")
    
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

        # --- Generator part ---
        # Fully connected layers
        self.ff_nets = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=1024),
            nn.BatchNorm1d(num_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=512),
            nn.ReLU(),
        )
        # Reshape layer (512 x 1 -> 256 x 1 x 2)
        self.reshape = nn.Unflatten(dim=1, unflattened_size=(256, 1, 2))
        # Transposed convolutional layers
        self.transp_conv_1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=(1, 2), stride=(1, 2)),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.transp_conv_2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=(1, 2), stride=(1, 2)),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.transp_conv_3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=(1, 2), stride=(1, 2)),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.transp_conv_4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=512, out_channels=1, kernel_size=(128, 1), stride=1),
            nn.Sigmoid() # Output values between 0 and 1
        )

        # --- Conditioner part ---
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=256, kernel_size=(128, 1), stride=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU()
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1, 2), stride=(1, 2)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU()
        )
        self.conv_3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1, 2), stride=(1, 2)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU()
        )
        self.conv_4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1, 2), stride=(1, 2)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU()
        )

    def forward(self, z, condition_matrix):
        # Conditioner process condition_matrix
        condition_step1 = self.conv_1(condition_matrix)
        condition_step2 = self.conv_2(condition_step1)
        condition_step3 = self.conv_3(condition_step2)
        condition_step4 = self.conv_4(condition_step3)

        # Generator process noise
        linear_output = self.ff_nets(z)
        reshaped_output = self.reshape(linear_output)
        
        # Generator Step 1: Input (Gen Start) + Condition (End)
        merged1 = torch.cat((reshaped_output, condition_step4), dim=1)
        output_step1 = self.transp_conv_1(merged1)
        # Generator Step 2: Input (Previous Output) + Condition (Previous Step)
        merged2 = torch.cat((output_step1, condition_step3), dim=1)
        output_step2 = self.transp_conv_2(merged2)
        # Generator Step 3: Input (Previous Output) + Condition (Previous Step)
        merged3 = torch.cat((output_step2, condition_step2), dim=1)
        output_step3 = self.transp_conv_3(merged3)
        # Generator Step 4: Input (Previous Output) + Condition (First Step)
        merged4 = torch.cat((output_step3, condition_step1), dim=1)
        output_step4 = self.transp_conv_4(merged4)
        
        return output_step4
    

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        # Convolutional layers
        self.first_conv = nn.Conv2d(in_channels=1, out_channels=14, kernel_size=(128, 2), stride=(1, 2)) # Separated by the other for the feature matching technique
        self.conv_layers = nn.Sequential(
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=14, out_channels=77, kernel_size=(1, 4), stride=(1, 2)),
            nn.BatchNorm2d(77),
            nn.LeakyReLU()
        )
        # Flatten layer
        self.flatten = nn.Flatten()
        # Feedforward layers
        self.ff_layers = nn.Sequential(
            nn.Linear(in_features=77*3, out_features=1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),
            nn.Linear(in_features=1024, out_features=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        f = self.first_conv(x)
        out_conv = self.conv_layers(f)
        out_flat = self.flatten(out_conv)
        output = self.ff_layers(out_flat)
        return output, f
    

class PianoGAN(BaseModel):
    def __init__(self, noise_dim: int = 100, learning_rate: float = 0.0002):
        """ 
        Initialize the base model. 
        Args:
            noise_dim: Dimension of the input noise vector for the generator.
            learning_rate: Learning rate for both generator and discriminator optimizers.
        """
        super().__init__(criterion=nn.BCEWithLogitsLoss(), learning_rate=learning_rate)
        
        self.save_hyperparameters()
        self.noise_dim = noise_dim
        
        # Sub-modules
        self.generator = Generator(input_size=noise_dim)
        self.discriminator = Discriminator()
        
        # Important: Disable automatic optimization to manage G and D separately
        self.automatic_optimization = False

    def forward(self, z: torch.Tensor, prev_bars: torch.Tensor) -> torch.Tensor:
        """
        Generate a new bar.
        Args:
            z: Noise vector
            prev_bars: Conditioning matrix (previous bar)
        """
        return self.generator(z, prev_bars)

    def configure_optimizers(self):
        """ Define the two separate optimizers for Discriminator and Generator. """

        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.learning_rate, betas=(0.5, 0.999))
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=self.learning_rate, betas=(0.5, 0.999))
        return [opt_d, opt_g], [] # standard Lightning format: (optimizers, schedulers)

    def training_step(self, batch, batch_idx):
        # Retrieve optimizers
        opt_d, opt_g = self.optimizers()
        
        # Unpacking the batch
        prev_bars, curr_bars = batch
        batch_size = prev_bars.size(0)
        
        # Generate noise
        # Note: Lightning automatically handles the device (self.device)
        noise = torch.randn(batch_size, self.noise_dim, device=self.device)

        # ---------------------
        # 1. Train Discriminator
        # ---------------------
        
        # Generazione fake (senza aggiornare gradienti G per ora)
        generated_bars = self.generator(noise, prev_bars)

        # Forward pass D su real
        real_output, _ = self.discriminator(curr_bars)
        
        # Forward pass D su fake (detach per non propagare su G)
        fake_output_detached, _ = self.discriminator(generated_bars.detach())

        # Calcolo Loss Discriminator (replicando `discriminator_loss` di test.py)
        # test.py usa BCEWithLogitsLoss.
        # real targets = 1, fake targets = 0
        real_loss = self.criterion(real_output, torch.ones_like(real_output))
        fake_loss = self.criterion(fake_output_detached, torch.zeros_like(fake_output_detached))
        d_loss = real_loss + fake_loss

        # Step di ottimizzazione Discriminator
        opt_d.zero_grad()
        self.manual_backward(d_loss)
        opt_d.step()

        # ---------------------
        # 2. Train Generator
        # ---------------------
        
        # Forward pass D su fake (questa volta serve il gradiente per G)
        fake_output, _ = self.discriminator(generated_bars)
        
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