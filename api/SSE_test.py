import json

from flask import Blueprint, Response, request, jsonify

from utils.api_utils import RESTfulAPI

SSE = Blueprint('SSE', __name__, url_prefix='/SSE')

REQUEST_DATA = {}


@SSE.route('/SSE', methods=['GET'])
async def SSE_test():
    global REQUEST_DATA

    def event_stream():
        yield json.dumps(dict(RESTfulAPI(code=200, status="success", data=request.json.get("id"))))

    return Response(event_stream(), mimetype="application/json")


@SSE.route('/data', methods=["POST"])
def add_data():
    global REQUEST_DATA
    REQUEST_DATA[request.json.get("id")] = request.json.get("data")
    return jsonify(code=200, status="success", message="successful add data"), 200
