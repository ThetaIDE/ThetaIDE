"""Legacy import alias; implementation lives in the bundled provider."""
from src.app.core.import_aliases import alias_module
alias_module(__name__, 'src.plugins.app.bundled.model_builder')
