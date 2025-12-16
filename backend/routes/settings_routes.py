from flask import Blueprint, jsonify, request
from pathlib import Path
import os
import json

bp = Blueprint('settings', __name__, url_prefix='/api/settings')

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