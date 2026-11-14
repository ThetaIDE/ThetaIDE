"""Hydra discovery adapter for the host's source-aware config resolver."""
from hydra.plugins.config_source import ConfigSource, ConfigResult
from hydra.core.object_type import ObjectType
from omegaconf import OmegaConf

from src.app.config_resolution import resolver_from_uri, composition_source
from hydra.plugins.search_path_plugin import SearchPathPlugin


class ThetaSearchPath(SearchPathPlugin):
    def manipulate_search_path(self, search_path):
        source = composition_source()
        if source is not None:
            search_path.prepend(provider="theta", path=source)


class ThetaConfigSource(ConfigSource):
    @staticmethod
    def scheme():
        return "theta"

    def __init__(self, provider, path):
        super().__init__(provider, path)
        self.resolver = resolver_from_uri(self.path)

    def load_config(self, config_path):
        resolved = self.resolver.resolve(config_path, required=True)
        text = resolved.file.read_text(encoding="utf-8")
        return ConfigResult(config=OmegaConf.create(text), path=self.full_path(), provider=self.provider,
                            header=self._get_header_dict(text))

    def is_config(self, config_path):
        return self.resolver.resolve(config_path) is not None if config_path else False

    def is_group(self, config_path):
        prefix = config_path.rstrip("/") + "/" if config_path else ""
        return any(name.startswith(prefix) for name in self.resolver.selectors())

    def available(self):
        return True

    def list(self, config_path, results_filter):
        prefix = config_path.rstrip("/") + "/" if config_path else ""
        results = set()
        for name in self.resolver.selectors():
            if not name.startswith(prefix):
                continue
            child = name[len(prefix):]
            if "/" in child:
                if results_filter in (None, ObjectType.GROUP): results.add(child.split("/", 1)[0])
            elif results_filter in (None, ObjectType.CONFIG):
                results.add(child.rsplit(".", 1)[0])
        return sorted(results)
