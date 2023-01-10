import torch

# Define your model, loss function, and optimizer
model = ...
criterion = ...
optimizer = ...

# Define a variable to keep track of the best validation loss
best_val_loss = float('inf')

# Define a variable to keep track of the number of consecutive epochs without improvement
early_stopping_count = 0

# Define the threshold for early stopping
early_stopping_threshold = 10

# Train the model
for epoch in range(num_epochs):

    # Train the model for one epoch
    ...

    # Evaluate the model on the validation set
    val_loss = ...

    # Check if the validation loss has improved
    if val_loss < best_val_loss:
        # Update the best validation loss
        best_val_loss = val_loss
        # Save the model parameters
        torch.save(model.state_dict(), 'best_model.pth')
        # Reset the early stopping count
        early_stopping_count = 0
    else:
        # Increment the early stopping count
        early_stopping_count += 1
        
    if early_stopping_count >= early_stopping_threshold:
        # The validation loss has not improved for `early_stopping_threshold` consecutive epochs, so stop training
        print("Early stopping!")
        break
