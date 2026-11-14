"""An independent tuner contributes setup and conditional fields through metadata."""
class Tuner:
    api_version = 1
    def settings(self):
        return [
            {'key': 'token', 'type': 'text', 'default': '', 'label': 'Connection'},
            {'key': 'attempts', 'type': 'int', 'default': 3, 'label': 'Attempts'},
            {'key': 'mode', 'type': 'choice', 'default': 'new', 'choices': ['new', 'resume']},
            {'key': 'reference', 'type': 'text', 'default': '', 'optional': True,
             'required_when': {'mode': 'resume'}},
        ]
    def validate_config(self, cfg): pass
    def validate_dimension(self, target, dimension): pass
    def compile(self, overrides, **context): return overrides
    def job_entrypoint(self, cfg, method_cfg, script, python): return script, python
    def study_name(self, cfg, method): return 'fixture'
    def prepare(self, cfg, storage, study): pass
    def finalize(self, group, experiment, method, storage, study): pass
    def storage(self, cfg): return None
