from pathlib import Path
import json

version = "0.0.4"

class PaintSystem:
    def __init__(self):
        self.appname = "AIPainter"
        self.dirname = {
            "Models": "models",
            "User": "user",
            "MMPose": "mmpose",
            "Log": "log",
            "Input": "input",
            "Output": "output",
            "Temp": "temp",
        }
        self.dirname_models = {
            "Checkpoint": "checkpoint",
            "VAE": "vae",
            "Lora": "lora",
        }
        self.suffix = {
            "System": ["json"],
            "Models": ["safetensors"],
            "Image": ["png", "jpeg"],
        }
        self.filename = {
            "System": f"{self.appname}.{self.suffix["System"][0]}"
        }
        self.systempath = {
            "Home": Path.home(),
            "Root": Path.home().joinpath(self.appname),
            "System": Path.home().joinpath(self.appname, self.filename["System"]),
        }
        self.system = {
            "Version": self.get_version(),
            "Checkpoint": {
                "checkpoint":"",
                "vae":"",
                "lora": {
                    "file": "",
                    "lora_dir":"",
                    "lora_name": "",
                    "lora_weight": 0.00,
                },
            },
            "ControlNet": {
                "canny":"",
                "ip2p":"",
                "inpaint":"",
                "mlsd":"",
                "depth":"",
                "normalbae":"",
                "seg":"",
                "lineart":"",
                "lineart_anime":"",
                "openpose":"",
                "scribble":"",
                "softedge":"",
                "shuffle":"",
                "tile":"",
            },
            "UperNet":{
                "seg":"",
            },
            "DefaultPrompts": {
                "Positive":"",
                "Negative":"",
            },
        }
        self.confpath = self.generate_confpath()
        self.modelspath = self.generate_modelspath()
        self.systemconf = self.system.copy()

    def generate_confpath(self):
        # confpathを生成する
        return {key: self.systempath["Root"].joinpath(value) for key, value in self.dirname.items()}
    
    def generate_modelspath(self):
        # modelspathを生成する
        return {key: self.confpath["Models"].joinpath(value) for key, value in self.dirname_models.items()}

    def create_directory(self):
        # rootディレクトリの作成
        rootpath = self.systempath["Root"]
        if not rootpath.exists():
            rootpath.mkdir()

        # confディレクトリの作成
        for path in self.confpath.values():
            if not path.exists():
                path.mkdir()
        
        # modelsディレクトリの作成
        for path in self.modelspath.values():
            if not path.exists():
                path.mkdir()

    def conf_system_checkpoint(self, conf:dict=None):
        if conf:
            self.systemconf["Checkpoint"] = conf
        
        return self.systemconf["Checkpoint"]
    
    def conf_system_controlnet(self, conf:dict=None):
        if conf:
            self.systemconf["ControlNet"] = conf

        return self.systemconf["ControlNet"]
    
    def conf_system_upernet(self, conf:dict=None):
        if conf:
            self.systemconf["UperNet"] = conf
        
        return self.systemconf["UperNet"]

    def conf_system_defaultprompts(self, conf:dict=None):
        if conf:
            self.systemconf["DefaultPrompts"] = conf

        return self.systemconf["DefaultPrompts"]

    def save_systemconf(self):

        # 設定ファイルを保存する
        data = self.systemconf
        filepath = str(self.systempath["System"])
        with open(filepath, 'w', encoding='utf-8') as f:
            try:
                json.dump(data, f, ensure_ascii=False, indent=4)
            except:
                # データが破損していた場合は空を保存
                data = self.system
                json.dump(data, f, ensure_ascii=False, indent=4)

    def load_systemconf(self):
        # 設定ファイルを読み込む
        data = None
        if self.systempath["System"].exists():
            filepath = str(self.systempath["System"])
            with open(filepath, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                    self.systemconf = data.copy()
                except json.JSONDecodeError:
                    # ファイルが破損していた場合は作成しなおす
                    self.systemconf = self.system.copy()
                    self.save_systemconf()           
        return data
    
    def get_version(self) -> str:
        return version
    
    def __call__(self):
        data = self.load_systemconf()    
        if not data:
            self.create_directory()
            self.save_systemconf()
            return None
        return data
