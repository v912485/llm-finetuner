from flask import Blueprint, jsonify, request
from pathlib import Path
import os
import json
from utils.auth import is_request_authenticated

bp = Blueprint('settings', __name__, url_prefix='/api/settings')


def _update_env_value(lines, key, value):
    updated = False
    for i, line in enumerate(lines):
        if line.startswith(f"{key}="):
            if value is None:
                lines.pop(i)
            else:
                lines[i] = f"{key}={value}\n"
            updated = True
            break
    if not updated and value is not None:
        lines.append(f"{key}={value}\n")
    return lines


def _save_env_values(updates):
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        with open(env_path, 'r') as f:
            lines = f.readlines()
    else:
        lines = []
    for key, value in updates.items():
        lines = _update_env_value(lines, key, value)
    with open(env_path, 'w') as f:
        f.writelines(lines)

@bp.route('/huggingface_token', methods=['GET'])
def get_huggingface_token():
    return jsonify({
        'status': 'success',
        'configured': bool(os.environ.get('HUGGING_FACE_TOKEN'))
    })

@bp.route('/huggingface_token', methods=['POST'])
def save_huggingface_token():
    data = request.json
    token = data.get('token')
    
    if not token:
        return jsonify({
            'status': 'error',
            'message': 'Token is required'
        }), 400
    
    try:
        env_path = Path(__file__).parent.parent / '.env'
        if env_path.exists():
            with open(env_path, 'r') as f:
                lines = f.readlines()
            
            token_updated = False
            for i, line in enumerate(lines):
                if line.startswith('HUGGING_FACE_TOKEN='):
                    lines[i] = f'HUGGING_FACE_TOKEN={token}\n'
                    token_updated = True
                    break
            
            if not token_updated:
                lines.append(f'HUGGING_FACE_TOKEN={token}\n')
            
            with open(env_path, 'w') as f:
                f.writelines(lines)
        else:
            with open(env_path, 'w') as f:
                f.write(f'HUGGING_FACE_TOKEN={token}\n')
        
        os.environ['HUGGING_FACE_TOKEN'] = token
        
        return jsonify({
            'status': 'success',
            'message': 'Token saved successfully'
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500 


@bp.route('/admin_token', methods=['GET'])
def get_admin_token_status():
    return jsonify({
        'status': 'success',
        'configured': bool(os.environ.get('ADMIN_TOKEN'))
    })


@bp.route('/admin_token', methods=['POST'])
def save_admin_token():
    data = request.json or {}
    token = data.get('token')

    if not token:
        return jsonify({
            'status': 'error',
            'message': 'Token is required'
        }), 400

    admin_configured = bool(os.environ.get('ADMIN_TOKEN'))
    if admin_configured and not is_request_authenticated():
        return jsonify({"status": "error", "message": "Unauthorized"}), 401

    try:
        env_path = Path(__file__).parent.parent / '.env'
        if env_path.exists():
            with open(env_path, 'r') as f:
                lines = f.readlines()

            token_updated = False
            for i, line in enumerate(lines):
                if line.startswith('ADMIN_TOKEN='):
                    lines[i] = f'ADMIN_TOKEN={token}\n'
                    token_updated = True
                    break

            if not token_updated:
                lines.append(f'ADMIN_TOKEN={token}\n')

            with open(env_path, 'w') as f:
                f.writelines(lines)
        else:
            with open(env_path, 'w') as f:
                f.write(f'ADMIN_TOKEN={token}\n')

        os.environ['ADMIN_TOKEN'] = token

        return jsonify({
            'status': 'success',
            'message': 'Admin token saved successfully'
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@bp.route('/config', methods=['GET'])
def get_config():
    try:
        config_path = Path(__file__).parent.parent / 'config.json'
        with open(config_path, 'r') as f:
            return jsonify(json.load(f))
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error loading config: {str(e)}'
        }), 500 


@bp.route('/config', methods=['POST'])
def update_config():
    data = request.json or {}
    ollama = data.get("ollama")
    if not isinstance(ollama, dict):
        return jsonify({
            "status": "error",
            "message": "ollama config is required"
        }), 400

    host = (ollama.get("host") or "").strip()
    port = ollama.get("port")
    if not host:
        return jsonify({"status": "error", "message": "ollama.host is required"}), 400
    try:
        port = int(port)
    except (TypeError, ValueError):
        return jsonify({"status": "error", "message": "ollama.port must be an integer"}), 400
    if port <= 0 or port > 65535:
        return jsonify({"status": "error", "message": "ollama.port is out of range"}), 400

    try:
        config_path = Path(__file__).parent.parent / "config.json"
        with open(config_path, "r") as f:
            config = json.load(f)
        if not isinstance(config, dict):
            config = {}
        config["ollama"] = {"host": host, "port": port}
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        return jsonify({"status": "success", "ollama": config["ollama"]})
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Error updating config: {str(e)}"
        }), 500


@bp.route('/gguf_config', methods=['GET'])
def get_gguf_config():
    return jsonify({
        "status": "success",
        "llama_cpp_dir": os.environ.get("LLAMA_CPP_DIR", ""),
        "gguf_converter_path": os.environ.get("GGUF_CONVERTER_PATH", ""),
        "gguf_outtype": os.environ.get("GGUF_OUTTYPE", "f16")
    })


@bp.route('/gguf_config', methods=['POST'])
def update_gguf_config():
    data = request.json or {}
    admin_configured = bool(os.environ.get('ADMIN_TOKEN'))
    if admin_configured and not is_request_authenticated():
        return jsonify({"status": "error", "message": "Unauthorized"}), 401

    llama_cpp_dir = (data.get("llama_cpp_dir") or "").strip()
    gguf_converter_path = (data.get("gguf_converter_path") or "").strip()
    gguf_outtype = (data.get("gguf_outtype") or "").strip()

    updates = {
        "LLAMA_CPP_DIR": llama_cpp_dir or None,
        "GGUF_CONVERTER_PATH": gguf_converter_path or None,
        "GGUF_OUTTYPE": gguf_outtype or None
    }

    try:
        _save_env_values(updates)
        if llama_cpp_dir:
            os.environ["LLAMA_CPP_DIR"] = llama_cpp_dir
        else:
            os.environ.pop("LLAMA_CPP_DIR", None)
        if gguf_converter_path:
            os.environ["GGUF_CONVERTER_PATH"] = gguf_converter_path
        else:
            os.environ.pop("GGUF_CONVERTER_PATH", None)
        if gguf_outtype:
            os.environ["GGUF_OUTTYPE"] = gguf_outtype
        else:
            os.environ.pop("GGUF_OUTTYPE", None)

        return jsonify({
            "status": "success",
            "llama_cpp_dir": os.environ.get("LLAMA_CPP_DIR", ""),
            "gguf_converter_path": os.environ.get("GGUF_CONVERTER_PATH", ""),
            "gguf_outtype": os.environ.get("GGUF_OUTTYPE", "f16")
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Error updating GGUF config: {str(e)}"
        }), 500