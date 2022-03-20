from typing import Type, Dict

from io import StringIO
from pydantic import BaseModel
from pydantic_yaml import YamlModel


class Sim2Config(YamlModel):
    class Config:
        default_files = ["test.yml"]
        extra = "allow"

    def yaml_desc(self, header=True, comments=True) -> str:
        """
        Dumps initial config in YAML but keeps class and Field descriptions as comments.
        """
        import ruamel.yaml

        yaml = ruamel.yaml.YAML()
        yaml.representer.add_representer(
            type(None),
            lambda self, d: self.represent_scalar("tag:yaml.org,2002:null", "~"),
        )
        yaml_str = StringIO()
        yaml.dump(self.dict(), stream=yaml_str)
        yaml_str.seek(0)
        dict_from_yaml = yaml.load(yaml_str)

        def yaml_model_printer(yaml_dict, parent):
            if parent.__doc__ and header:
                yaml_dict.yaml_set_start_comment(parent.__doc__ + "\n")

            for k in yaml_dict.keys():
                key_val = getattr(parent, k)
                if isinstance(key_val, BaseModel):
                    yaml_dict.yaml_set_comment_before_after_key(k, before="\n")
                    yaml_model_printer(yaml_dict[k], key_val)
                elif parent.__fields__[k].field_info.description and comments:
                    yaml_dict.yaml_set_comment_before_after_key(
                        k, before=parent.__fields__[k].field_info.description
                    )

        yaml_model_printer(dict_from_yaml, self)

        yaml_str = StringIO()
        yaml.dump(dict_from_yaml, yaml_str)
        yaml_str.seek(0)
        return yaml_str.read()
