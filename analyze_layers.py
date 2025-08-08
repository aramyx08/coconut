import torch
import torch.nn.functional as F
import torch.distributed as dist
from utils import LayeredResidualCollector

def find_best_steering_layer(model, dataloader, layers_to_test, device='cuda'):
    """
    Finds the best layer for steering by finding the one where the mean hidden states
    for correct and incorrect reasoning are most dissimilar.

    Args:
        model: The model to analyze. It should be configured to output hidden states.
        dataloader: A dataloader that yields batches of (correct_input, incorrect_input).
        layers_to_test: A list of layer indices to check (e.g., [6, 8, 10]).
        device: The device to run the model on.

    Returns:
        A tuple of (best_layer_index, best_steering_vector) on rank 0.
        Returns (None, None) on other ranks.
    """
    # This collector is now layer-aware
    collector = LayeredResidualCollector(device=device)

    print("--- Step 1: Collecting hidden states for all layers ---")
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            # Assuming batch gives you paired correct/incorrect examples
            # This part may need to be adapted based on your actual dataloader structure
            correct_ids, incorrect_ids = batch
            correct_ids, incorrect_ids = correct_ids.to(device), incorrect_ids.to(device)

            # --- IMPORTANT ---
            # The model's forward pass must be called with a parameter that ensures
            # it returns all hidden states. For most Hugging Face models, this is
            # `output_hidden_states=True`.
            correct_outputs = model(correct_ids, output_hidden_states=True)
            incorrect_outputs = model(incorrect_ids, output_hidden_states=True)

            # The 'hidden_states' output is a tuple where the Nth element corresponds
            # to the output of the Nth layer. Index 0 is the initial embeddings.
            for layer in layers_to_test:
                # Extract the hidden state for the last token for the current layer
                correct_state = correct_outputs.hidden_states[layer][:, -1, :]
                incorrect_state = incorrect_outputs.hidden_states[layer][:, -1, :]

                # Add these states to our new collector
                collector.add_state(correct_state, is_correct=True, layer_index=layer)
                collector.add_state(incorrect_state, is_correct=False, layer_index=layer)

            if i % 10 == 0:
                print(f"Processed batch {i+1}/{len(dataloader)}")

    # Gather all collected states from the distributed processes to rank 0
    collector._gather_states()

    print("\n--- Step 2: Finding layer with least cosine similarity ---")
    best_layer = -1
    min_similarity = float('inf')
    best_steering_vector = None

    # This calculation should only be done on the main process (rank 0)
    # to avoid redundant computations and printing.
    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank == 0:
        for layer in layers_to_test:
            # Get the mean vectors for this layer
            residual, good_mean, bad_mean = collector.compute_residual_for_layer(layer)

            if residual is None:
                # This happens if there weren't enough samples for this layer
                continue

            # Calculate the cosine similarity between the two mean vectors
            similarity = F.cosine_similarity(good_mean.unsqueeze(0), bad_mean.unsqueeze(0))

            print(f"Layer {layer}: Cosine Similarity = {similarity.item():.4f}")

            if similarity < min_similarity:
                min_similarity = similarity
                best_layer = layer
                best_steering_vector = residual

        print(f"\n--- Result ---")
        if best_layer != -1:
            print(f"Best layer found: {best_layer} (with cosine similarity: {min_similarity:.4f})")
        else:
            print("Could not determine best layer. Not enough samples collected.")

        return best_layer, best_steering_vector

    # Other ranks that are not rank 0 will return None
    return None, None
