class Callbacks:
    _callbacks = {
        'on_pretrain_routine_start': [],
        'on_pretrain_routine_end': [],
        'on_train_start': [],
        'on_train_epoch_start': [],
        'on_train_batch_start': [],
        'optimizer_step': [],
        'on_before_zero_grad': [],
        'on_train_batch_end': [],
        'on_train_epoch_end': [],
        'on_val_start': [],
        'on_val_batch_start': [],
        'on_val_image_end': [],
        'on_val_batch_end': [],
        'on_val_end': [],
        'on_fit_epoch_end': [],
        'on_model_save': [],
        'on_train_end': [],
        'update_keys':[],
        'teardown': [],
    }
    def __init__(self):
        return
    def register_action(self, hook, name='', callback=None):
        assert hook in self._callbacks, f"hook '{hook}' not found in callbacks {self._callbacks}"
        assert callable(callback), f"callback '{callback}' is not callable"
        self._callbacks[hook].append({'name': name, 'callback': callback})
    def get_registered_actions(self, hook=None):
        if hook:
            return self._callbacks[hook]
        else:
            return self._callbacks
    def run_callbacks(self, hook, *args, **kwargs):
        for logger in self._callbacks[hook]:
            logger['callback'](*args, **kwargs)
    def on_pretrain_routine_start(self, *args, **kwargs):
        self.run_callbacks('on_pretrain_routine_start', *args, **kwargs)
    def on_pretrain_routine_end(self, *args, **kwargs):
        self.run_callbacks('on_pretrain_routine_end', *args, **kwargs)
    def on_train_start(self, *args, **kwargs):
        self.run_callbacks('on_train_start', *args, **kwargs)
    def on_train_epoch_start(self, *args, **kwargs):
        self.run_callbacks('on_train_epoch_start', *args, **kwargs)
    def on_train_batch_start(self, *args, **kwargs):
        self.run_callbacks('on_train_batch_start', *args, **kwargs)
    def optimizer_step(self, *args, **kwargs):
        self.run_callbacks('optimizer_step', *args, **kwargs)
    def on_before_zero_grad(self, *args, **kwargs):
        self.run_callbacks('on_before_zero_grad', *args, **kwargs)
    def on_train_batch_end(self, *args, **kwargs):
        self.run_callbacks('on_train_batch_end', *args, **kwargs)
    def on_train_epoch_end(self, *args, **kwargs):
        self.run_callbacks('on_train_epoch_end', *args, **kwargs)
    def on_val_start(self, *args, **kwargs):
        self.run_callbacks('on_val_start', *args, **kwargs)
    def on_val_batch_start(self, *args, **kwargs):
        self.run_callbacks('on_val_batch_start', *args, **kwargs)
    def on_val_image_end(self, *args, **kwargs):
        self.run_callbacks('on_val_image_end', *args, **kwargs)
    def on_val_batch_end(self, *args, **kwargs):
        self.run_callbacks('on_val_batch_end', *args, **kwargs)
    def on_val_end(self, *args, **kwargs):
        self.run_callbacks('on_val_end', *args, **kwargs)
    def on_fit_epoch_end(self, *args, **kwargs):
        self.run_callbacks('on_fit_epoch_end', *args, **kwargs)
    def on_model_save(self, *args, **kwargs):
        self.run_callbacks('on_model_save', *args, **kwargs)
    def on_train_end(self, *args, **kwargs):
        self.run_callbacks('on_train_end', *args, **kwargs)
    def teardown(self, *args, **kwargs):
        self.run_callbacks('teardown', *args, **kwargs)