#!/usr/bin/env python3
"""
Custom Node ComfyUI para upload de pastas para Hugging Face
"""

import os
import sys
import zipfile
import tempfile
import subprocess
from pathlib import Path
from datetime import datetime
import logging
from typing import Optional, Tuple, Any
import torch

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def install_dependencies():
    """Instala as dependências necessárias"""
    try:
        from huggingface_hub import HfApi, upload_file
        return HfApi, upload_file
    except ImportError:
        logger.info("📦 Instalando huggingface_hub...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub", "--quiet"])
        from huggingface_hub import HfApi, upload_file
        return HfApi, upload_file


def find_folder(folder_path: str) -> Optional[str]:
    """Procura pela pasta, adicionando '/' no início se necessário"""
    possible_paths = []
    
    if folder_path.startswith('/'):
        possible_paths.append(folder_path)
    else:
        possible_paths.append(f"/{folder_path}")
        possible_paths.append(folder_path)
    
    possible_paths.extend([
        f"./{folder_path}",
        f"../{folder_path}",
        f"/workspace/{folder_path}" if not folder_path.startswith('/workspace') else folder_path
    ])
    
    print(f"🔍 Procurando pasta: {folder_path}")
    
    for path in possible_paths:
        if os.path.isdir(path):
            real_path = os.path.realpath(path)
            print(f"✅ Pasta encontrada: {real_path}")
            return real_path
    
    print(f"❌ Pasta não encontrada!")
    print("Caminhos tentados:")
    for path in possible_paths:
        print(f"  - {path}")
    
    return None


def create_zip(folder_path: str, zip_path: str) -> bool:
    """Cria um arquivo ZIP da pasta especificada"""
    try:
        print(f"📦 Compactando pasta: {folder_path}")
        
        p_folder_path = Path(folder_path)
        all_files = list(p_folder_path.rglob('*'))
        total_files = sum(1 for f in all_files if f.is_file())
        
        if total_files == 0:
            print("⚠️  A pasta parece estar vazia!")
            return False
        
        print(f"📊 Total de arquivos a processar: {total_files}")
        processed_files = 0
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zipf:
            for file_path in all_files:
                if file_path.is_file():
                    arcname = file_path.relative_to(p_folder_path)
                    zipf.write(file_path, arcname)
                    processed_files += 1
                    
                    if processed_files % 50 == 0 or processed_files == total_files:
                        progress = (processed_files / total_files) * 100
                        print(f"  ⏳ Progresso ZIP: {progress:.1f}% ({processed_files}/{total_files} arquivos)")
                
                elif file_path.is_dir() and not list(file_path.iterdir()):
                    arcname = file_path.relative_to(p_folder_path)
                    zipf.write(file_path, arcname)
        
        zip_size = os.path.getsize(zip_path) / (1024*1024)
        print(f"✅ ZIP criado com sucesso!")
        print(f"📏 Tamanho: {zip_size:.2f} MB")
        print(f"📄 Arquivos processados: {processed_files}")
        return True
        
    except Exception as e:
        print(f"❌ Erro ao criar ZIP: {e}")
        return False


def upload_to_hf(zip_path: str, repo_id: str, token: str, HfApi, upload_file) -> Optional[str]:
    """Faz upload do arquivo ZIP para o Hugging Face"""
    print(f"🚀 Iniciando upload para: {repo_id}")
    
    filename = os.path.basename(zip_path)
    
    try:
        api = HfApi(token=token)
        
        # Verifica/cria repositório
        try:
            api.repo_info(repo_id=repo_id, repo_type="model", token=token)
            print(f"📂 Repositório encontrado: {repo_id}")
        except Exception:
            print(f"🆕 Criando repositório: {repo_id}")
            api.create_repo(repo_id=repo_id, repo_type="model", token=token, private=False)
        
        # Upload
        print(f"📤 Fazendo upload: {filename}")
        print("⏳ Upload em progresso... (pode demorar dependendo do tamanho)")
        
        url = upload_file(
            path_or_fileobj=zip_path,
            path_in_repo=filename,
            repo_id=repo_id,
            token=token,
            repo_type="model",
            commit_message=f"Upload automático: {filename}"
        )
        
        print(f"✅ Upload concluído com sucesso!")
        print(f"🔗 URL: {url}")
        return url
        
    except Exception as e:
        print(f"❌ Erro no upload: {e}")
        return None


def validate_token(token: str) -> bool:
    """Valida o token do Hugging Face"""
    if not token:
        print("❌ Token não fornecido!")
        return False
    
    if not token.startswith("hf_"):
        print("⚠️  Token pode estar inválido (deveria começar com 'hf_')")
    
    if len(token) < 20:
        print("⚠️  Token parece muito curto")
    
    return True


def validate_repo(repo: str) -> bool:
    """Valida o formato do repositório"""
    if not repo:
        print("❌ Repositório não fornecido!")
        return False
    
    if "/" not in repo:
        print("❌ Repositório deve estar no formato 'usuario/nome'")
        return False
    
    parts = repo.split("/")
    if len(parts) != 2 or not parts[0] or not parts[1]:
        print("❌ Formato de repositório inválido!")
        return False
    
    return True


def generate_zip_name(folder_path: str, custom_name: str = None) -> str:
    """Gera o nome do arquivo ZIP"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if custom_name and custom_name.strip():
        if custom_name.lower().endswith('.zip'):
            custom_name = custom_name[:-4]
        return f"{custom_name}_{timestamp}.zip"
    else:
        folder_name = Path(folder_path).name
        return f"{folder_name}_{timestamp}.zip"


class HuggingFaceUploadNode:
    """
    Custom Node para fazer upload de pastas para Hugging Face
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "hf_token": ("STRING", {
                    "multiline": False,
                    "default": "hf_your_token_here"
                }),
                "repo_id": ("STRING", {
                    "multiline": False,
                    "default": "usuario/repo"
                }),
                "folder_path": ("STRING", {
                    "multiline": False,
                    "default": "/workspace/PACKS_CRIADOS/test/pack/hestia"
                }),
                "zip_name": ("STRING", {
                    "multiline": False,
                    "default": ""
                }),
                "trigger_image_1": ("IMAGE",),
                "trigger_image_2": ("IMAGE",),
                "trigger_image_3": ("IMAGE",),
                "trigger_image_4": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("download_url",)
    FUNCTION = "upload_folder"
    CATEGORY = "upload"
    
    def upload_folder(self, hf_token, repo_id, folder_path, zip_name, trigger_image_1, trigger_image_2, trigger_image_3):
        """
        Função principal que executa o upload
        As imagens são apenas triggers, não são usadas
        """
        
        # Banner
        print("🎯 Packreator ComfyUI Upload Tool v2.0")
        print("=" * 50)
        
        # Instala dependências
        try:
            HfApi, upload_file = install_dependencies()
        except Exception as e:
            error_msg = f"❌ Erro ao instalar dependências: {e}"
            print(error_msg)
            return (error_msg,)
        
        # Validações
        if not validate_token(hf_token):
            return ("❌ Token inválido",)
        
        if not validate_repo(repo_id):
            return ("❌ Repositório inválido",)
        
        # Procura a pasta
        folder_path = find_folder(folder_path)
        if not folder_path:
            return ("❌ Pasta não encontrada",)
        
        # Gera nome do ZIP
        zip_filename = generate_zip_name(folder_path, zip_name)
        
        print(f"📋 Configurações:")
        print(f"  🗂️  Pasta: {folder_path}")
        print(f"  📦 ZIP: {zip_filename}")
        print(f"  🎯 Destino: {repo_id}")
        print(f"  🔑 Token: {hf_token[:10]}...")
        
        # Processa
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                zip_path = os.path.join(temp_dir, zip_filename)
                
                # Cria ZIP
                if not create_zip(folder_path, zip_path):
                    return ("❌ Falha ao criar ZIP",)
                
                # Upload
                upload_url = upload_to_hf(zip_path, repo_id, hf_token, HfApi, upload_file)
                
                if upload_url:
                    success_msg = f"✅ Upload concluído! URL: {upload_url}"
                    print("\n🎉 PROCESSO CONCLUÍDO COM SUCESSO!")
                    print("=" * 50)
                    print(f"📦 Arquivo: {zip_filename}")
                    print(f"🌐 Repositório: {repo_id}")
                    print(f"🔗 URL: {upload_url}")
                    print("=" * 50)
                    return (upload_url,)
                else:
                    return ("❌ Falha no upload",)
                    
        except Exception as e:
            error_msg = f"❌ Erro durante o processo: {e}"
            print(error_msg)
            return (error_msg,)


# Mapeamento dos nós
NODE_CLASS_MAPPINGS = {
    "HuggingFaceUploadNode": HuggingFaceUploadNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HuggingFaceUploadNode": "🤗 HuggingFace Upload Packreator"
}
