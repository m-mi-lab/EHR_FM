class MOEManager:
    """
    Basic wrapper class for tracking, storing, and aggregating auxiliary
    losses across multiple MoE layers in the model.
    
    Note: Losses are stored as tensors (not detached) because they need to
    participate in backpropagation. The reset methods MUST be called after
    each forward pass to prevent memory accumulation.
    """

    def __init__(self):
        self.aux_loss = []
        self.router_z_loss = []
        self._max_accumulated_losses = 100  # Safety limit to detect leaks
    
    def reset_aux_loss(self):
        # Clear the list to allow garbage collection
        self.aux_loss.clear()
        self.aux_loss = []
    
    def reset_router_z_loss(self):
        # Clear the list to allow garbage collection
        self.router_z_loss.clear()
        self.router_z_loss = []
    
    def reset_all(self):
        """Reset all accumulated losses. Call this at the start of each forward pass as a safety measure."""
        self.reset_aux_loss()
        self.reset_router_z_loss()
    
    def add_aux_loss(self, loss):
        if len(self.aux_loss) >= self._max_accumulated_losses:
            # Safety check: if too many losses accumulated, something is wrong
            import warnings
            warnings.warn(
                f"MOEManager: aux_loss has {len(self.aux_loss)} accumulated entries. "
                "This may indicate a memory leak. Resetting to prevent OOM.",
                RuntimeWarning
            )
            self.reset_aux_loss()
        self.aux_loss.append(loss)
    
    def add_router_z_loss(self, loss):
        if len(self.router_z_loss) >= self._max_accumulated_losses:
            # Safety check: if too many losses accumulated, something is wrong
            import warnings
            warnings.warn(
                f"MOEManager: router_z_loss has {len(self.router_z_loss)} accumulated entries. "
                "This may indicate a memory leak. Resetting to prevent OOM.",
                RuntimeWarning
            )
            self.reset_router_z_loss()
        self.router_z_loss.append(loss)
    
    def aggregate_aux_loss(self):
        if not self.aux_loss:
            return None
        return sum(self.aux_loss)

    def aggregate_router_z_loss(self):
        if not self.router_z_loss:
            return None
        return sum(self.router_z_loss)

MANAGER = MOEManager()