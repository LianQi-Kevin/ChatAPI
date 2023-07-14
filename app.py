import json
import logging

from flask import Flask
from flask_cors import CORS

# APIs
from api.ChatGLM2_6B import ChatGLM2
# Utils
from utils.logging_utils import log_set

"""Create Flask Application."""
# init FlaskAPI
app = Flask(__name__)
# Allow CORS
CORS(app)
# logging
log_set(logging.DEBUG, log_save=False)
# register_blueprint
app.register_blueprint(ChatGLM2)

if __name__ == '__main__':
    with open("./config.json") as f:
        server_configs = json.loads("".join(f.readlines()))["server"]
    app.run(
        port=server_configs["port"],
        debug=server_configs["debug"],
        host=server_configs["host"]
    )
