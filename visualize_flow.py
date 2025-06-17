import torch
import torchvision
from torchvision import transforms
from omegaconf import OmegaConf, DictConfig
import matplotlib.pyplot as plt
import numpy as np
import math # 确保导入 math

# 从你的项目中导入必要的模块
from utils_train import seed_everything, make_config, make_bfn, worker_init_function
from data import make_datasets # 假设 make_datasets 和 bin_mnist_transform 在 data.py 中
from model import BFN

def visualize_bayesian_flow(
    config_path: str,
    model_checkpoint_path: str,
    image_indices: list, # 要可视化的测试集图像的索引列表
    num_time_steps: int = 20,
    max_time: float = 1/3,
    save_dir: str = "./flow_visualizations"
):
    """
    Generates visualizations similar to Figure 13 from the Bayesian Flow Networks paper.
    """
    seed_everything(42) # Use a fixed seed for reproducibility of image selection if needed

    # --- 1. Configuration and Model Loading ---
    cfg_file = config_path
    cli_conf = OmegaConf.create({"config_file": cfg_file, "load_model": model_checkpoint_path})
    
    # Get model and data config from the training config file
    # train_cfg is the main config object for model, data, etc.
    train_cfg = make_config(cli_conf.config_file)

    # Device setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load BFN model
    bfn_model: BFN = make_bfn(train_cfg.model) #
    
    # Load state dict
    # The model was saved by accelerator, so it might be the unwrapped model.
    # If train.py saves accelerator.save_state(checkpoint_dir),
    # then model is at checkpoint_dir/pytorch_model.bin (for non-EMA)
    # or checkpoint_dir/ema_model.pt (if you saved EMA separately)
    # Your command used pytorch_model.bin, so we assume it's the non-EMA model state.
    state_dict = torch.load(cli_conf.load_model, map_location="cpu")
    
    # If the model was saved via accelerator.save_state and you're loading the raw model,
    # it might be a DDP-wrapped model. Try to unwrap if necessary.
    # However, BFN is a nn.Module, so direct loading should work if not wrapped or if keys match.
    # If it's an EMA model, it's usually saved directly as state_dict.
    try:
        bfn_model.load_state_dict(state_dict)
    except RuntimeError as e:
        print(f"Standard load_state_dict failed: {e}. Attempting to load by removing 'module.' prefix (DDP)...")
        try:
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k # remove `module.`
                new_state_dict[name] = v
            bfn_model.load_state_dict(new_state_dict)
            print("Successfully loaded model after removing 'module.' prefix.")
        except Exception as e2:
            print(f"Failed to load model even after attempting to handle DDP prefix: {e2}")
            raise e

    bfn_model.to(device)
    bfn_model.eval()
    print("Model loaded successfully.")

    # --- 2. Data Loading ---
    # make_datasets returns train, val, test
    _, _, test_dataset = make_datasets(train_cfg.data) #
    print(f"Test dataset loaded. Size: {len(test_dataset)}")

    if not image_indices:
        print("No image indices provided. Exiting.")
        return

    for img_idx_in_list, selected_image_index in enumerate(image_indices):
        if selected_image_index >= len(test_dataset):
            print(f"Image index {selected_image_index} is out of bounds for test dataset size {len(test_dataset)}. Skipping.")
            continue
        
        print(f"Processing image at index: {selected_image_index}")
        # data_point is a tuple (image_tensor, label_scalar)
        image_data, image_label_scalar = test_dataset[selected_image_index]
        
        # Prepare batch (batch size 1)
        # Expected shape for MNIST in your BFN.bayesian_flow seems to be (B, H, W, C)
        # Your data.py's bin_mnist_transform returns (H, W, C)
        current_image_data_batch = image_data.unsqueeze(0).to(device) # Shape: [1, H, W, C]
        current_image_label_batch = torch.tensor([image_label_scalar], device=device) # Shape: [1]

        # --- 3. Time Steps ---
        time_points = torch.linspace(0, max_time, num_time_steps, device=device) #

        input_dist_sequence_imgs = []
        output_dist_sequence_imgs = []

        # --- 4. Iterate over time and compute distributions ---
        with torch.no_grad():
            for t_idx, t_val in enumerate(time_points):
                print(f"  Time step {t_idx + 1}/{num_time_steps}, t = {t_val.item():.4f}")

                # Reshape t for different parts of the model if necessary
                # For bayesian_flow, t can be shaped like data for broadcasting
                t_for_flow = t_val.expand_as(current_image_data_batch) # Shape: [1, H, W, C]

                # For UNetModel, t is typically (B,) or (B,1) and then processed
                # UNetModel.forward internally does: timesteps = t.flatten(start_dim=1)[:, 0] * 4000
                # So, a (B,1) tensor for t is appropriate here.
                t_for_net = t_val.view(1, 1) # Shape: [1, 1]

                # a. Calculate Input Distribution parameters
                # DiscreteBayesianFlow.forward(self, data: Tensor, label: Tensor, t: Tensor)
                input_params_tuple = bfn_model.bayesian_flow(
                    current_image_data_batch, 
                    current_image_label_batch, 
                    t_for_flow
                )
                # For binary MNIST (n_classes=2), input_params_tuple[0] contains probabilities P(pixel=1)
                # Shape is (B, H, W, 1)
                input_prob_image = input_params_tuple[0].squeeze(0).cpu().numpy() # Remove batch, move to CPU -> (H,W,1)
                input_dist_sequence_imgs.append(input_prob_image)

                # b. Calculate Output Distribution parameters
                net_inputs = bfn_model.bayesian_flow.params_to_net_inputs(input_params_tuple) #
                
                # bfn_model.net is UNetModel. UNetModel.forward takes (adapted_input, t, label)
                output_params_from_net = bfn_model.net(
                    net_inputs, 
                    t_for_net, # Pass the correctly shaped t for the network
                    current_image_label_batch
                )
                
                # Get distribution object from output parameters
                # For BernoulliFactory, output_params_from_net has shape (B, N_pixels, 1, 1) or similar
                # and get_dist expects logits.
                output_dist = bfn_model.loss.distribution_factory.get_dist(output_params_from_net)
                
                # For Bernoulli, .bernoulli.probs gives P(class=1)
                # The shape of output_dist.bernoulli.probs will be (B, N_pixels) if N_pixels = H*W
                output_prob_flat = output_dist.bernoulli.probs # Shape: [1, H*W]
                
                h, w, c = current_image_data_batch.shape[1:] # H, W, C from original image data
                output_prob_image = output_prob_flat.view(1, h, w, c).squeeze(0).cpu().numpy() # Reshape and remove batch -> (H,W,C)
                output_dist_sequence_imgs.append(output_prob_image)
        
        # --- 5. Visualization ---
        import os
        os.makedirs(save_dir, exist_ok=True)

        plot_image_matrix(
            input_dist_sequence_imgs, 
            time_points.cpu().numpy(),
            title=f"Input Distribution (Image Index: {selected_image_index}, Label: {image_label_scalar})",
            filename=os.path.join(save_dir, f"input_dist_img{selected_image_index}.png")
        )
        plot_image_matrix(
            output_dist_sequence_imgs,
            time_points.cpu().numpy(),
            title=f"Output Distribution (Image Index: {selected_image_index}, Label: {image_label_scalar})",
            filename=os.path.join(save_dir, f"output_dist_img{selected_image_index}.png")
        )
    print(f"Visualizations saved to {save_dir}")

def plot_image_matrix(image_sequence, time_values, title, filename, cols=5):
    """Helper function to plot a sequence of images in a grid."""
    num_images = len(image_sequence)
    rows = math.ceil(num_images / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
    axes = axes.flatten() # Flatten in case of single row/col

    for i, img_array in enumerate(image_sequence):
        ax = axes[i]
        # Assuming image_array is (H, W, 1) or (H, W)
        ax.imshow(img_array.squeeze(), cmap='gray', vmin=0, vmax=1)
        ax.set_title(f"t={time_values[i]:.3f}", fontsize=8)
        ax.axis('off')

    # Hide any unused subplots
    for j in range(num_images, len(axes)):
        axes[j].axis('off')

    fig.suptitle(title, fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make space for suptitle
    plt.savefig(filename)
    print(f"Saved plot: {filename}")
    plt.close(fig) # Close the figure to free memory


if __name__ == "__main__":
    # --- Configuration for the script ---
    CONFIG_PATH = "./configs/mnist_discrete_.yaml"  # Path to your model's training config
    MODEL_CHECKPOINT_PATH = "./checkpoints/BFN/last/ema_model.pt" # Path to your trained model
    
    # Select a few image indices from the MNIST test set to visualize
    IMAGE_INDICES_TO_VISUALIZE = [0, 7, 100] # Example indices, pick any you like
    
    NUM_TIME_STEPS_VIZ = 20  # Number of time steps to visualize (as in paper Fig 13)
    MAX_TIME_VIZ = 1/3       # Max time t for visualization (as in paper Fig 13)
    SAVE_DIRECTORY = "./flow_visualizations"

    visualize_bayesian_flow(
        config_path=CONFIG_PATH,
        model_checkpoint_path=MODEL_CHECKPOINT_PATH,
        image_indices=IMAGE_INDICES_TO_VISUALIZE,
        num_time_steps=NUM_TIME_STEPS_VIZ,
        max_time=MAX_TIME_VIZ,
        save_dir=SAVE_DIRECTORY
    )