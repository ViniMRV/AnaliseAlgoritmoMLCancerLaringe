#!/usr/bin/env python3
"""
Conversor automático de NSP para WAV focado na estrutura DadosDoencas.
Filtra apenas arquivos 'a_n.nsp' dentro de pastas 'vowels', organizando
a saída por nome da doença.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Tuple, List, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

def ensure_ffmpeg():
    from shutil import which
    if which("ffmpeg") is None:
        sys.exit("Erro: ffmpeg não encontrado no PATH. Instale o ffmpeg e tente novamente.")

def get_disease_name(file_path: Path, root_folder: Path) -> Optional[str]:
    """
    Tenta extrair o nome da doença baseado na estrutura relativa.
    Estrutura esperada: Root / NomeDoenca / ID / vowels / arquivo
    """
    try:
        # Pega o caminho relativo à pasta raiz (DadosDoencas)
        relative = file_path.relative_to(root_folder)
        parts = relative.parts
        
        # Se a estrutura for: Doenca/ID/vowels/arquivo, o nome da doença é a parte 0
        if len(parts) >= 4 and "vowels" in parts:
            return parts[0]
    except ValueError:
        pass
    return None

def iter_target_files(root: Path):
    """
    Varre recursivamente buscando apenas os arquivos alvo:
    - Extensão .nsp
    - Dentro de uma pasta 'vowels'
    - Nome terminando em 'a_n.nsp'
    """
    for p in root.rglob("*.nsp"):
        if not p.is_file():
            continue
            
        # Filtro 1: Deve estar numa pasta chamada "vowels"
        # Verificamos se 'vowels' é o nome da pasta pai imediata
        if p.parent.name != "vowels":
            continue

        # Filtro 2: O arquivo deve terminar com 'a_n.nsp'
        # Exemplo: 1472-a_n.nsp
        if not p.name.endswith("-a_n.nsp") and not p.name.endswith("_a_n.nsp"):
            # Adicionei checagem para '_' caso o padrão varie, mas foca no '-a_n'
            continue
            
        yield p

def convert_one(infile: str, outfile: str, resample: int | None, overwrite: bool, dry_run: bool) -> Tuple[str, str]:
    """
    Função Worker. Converte infile para outfile.
    """
    p_in = Path(infile)
    p_out = Path(outfile)

    if p_out.exists() and not overwrite:
        return ("skip_exists", infile)

    if dry_run:
        return ("dry_run", f"{p_in.name} -> {p_out}")

    # Garante que a pasta de destino da doença exista
    p_out.parent.mkdir(parents=True, exist_ok=True)

    # Comando ffmpeg
    cmd = [
        "ffmpeg",
        "-v", "error",        # apenas erros
        "-y" if overwrite else "-n",
        "-i", str(p_in),
        "-c:a", "pcm_s16le",
    ]
    if resample:
        cmd += ["-ar", str(resample)]
    cmd.append(str(p_out))

    try:
        completed = subprocess.run(cmd, capture_output=True, text=True)
        if completed.returncode == 0:
            return ("ok", infile)
        else:
            tail = (completed.stderr or "").strip().splitlines()[-1:] or [""]
            return ("fail", f"{infile} :: {tail[0]}")
    except Exception as e:
        return ("fail", f"{infile} :: {e}")

def main():
    parser = argparse.ArgumentParser(description="Conversor específico para estrutura DadosDoencas.")
    parser.add_argument("--root", type=Path, default=Path("DadosDoencas"), 
                        help="Pasta raiz contendo as doenças (Padrão: ./DadosDoencas)")
    parser.add_argument("--resample", type=int, default=None, help="Taxa de amostragem alvo (ex: 44100)")
    parser.add_argument("--overwrite", action="store_true", help="Sobrescrever arquivos existentes")
    parser.add_argument("--dry-run", action="store_true", help="Simular sem converter")
    parser.add_argument("--workers", type=int, default=0, help="Número de processos paralelos")
    
    args = parser.parse_args()

    ensure_ffmpeg()

    # Define diretórios
    root_input = args.root.resolve()
    script_dir = Path(__file__).parent.resolve()
    root_output = script_dir / "AudiosConvertidos"

    if not root_input.exists():
        sys.exit(f"Erro: Pasta de entrada não encontrada: {root_input}")

    print(f"Lendo arquivos de: {root_input}")
    print(f"Salvando em: {root_output}")

    # Coletar lista de tarefas (input_path, output_path)
    tasks = []
    
    # Lazy import tqdm
    try:
        from tqdm import tqdm
        use_bar = True
    except ImportError:
        use_bar = False

    print("Escaneando diretórios...")
    
    files_found = list(iter_target_files(root_input))
    
    for p in files_found:
        disease_name = get_disease_name(p, root_input)
        
        if not disease_name:
            print(f"Aviso: Não foi possível identificar a doença para {p}, pulando.")
            continue

        # Define o caminho de saída: AudiosConvertidos / NomeDoenca / NomeArquivo.wav
        # Mantemos o nome original (ex: 1472-a_n.wav) para preservar o ID.
        out_name = p.with_suffix(".wav").name
        out_path = root_output / disease_name / out_name
        
        tasks.append((str(p), str(out_path)))

    total = len(tasks)
    if total == 0:
        print("Nenhum arquivo correspondente (vowels/*-a_n.nsp) encontrado.")
        return

    print(f"Encontrados {total} arquivos para conversão.")

    # Configuração dos workers
    cpu_count = os.cpu_count() or 1
    workers = args.workers if args.workers and args.workers > 0 else cpu_count

    converted = 0
    skipped = 0
    failed = 0

    if use_bar:
        pbar = tqdm(total=total, unit="file", desc="Convertendo", ncols=80)

    with ProcessPoolExecutor(max_workers=workers) as ex:
        # Submete tarefas passando explicitamente o caminho de saída
        futures = [
            ex.submit(convert_one, inp, out, args.resample, args.overwrite, args.dry_run)
            for (inp, out) in tasks
        ]

        for fut in as_completed(futures):
            status, info = fut.result()
            if status == "ok":
                converted += 1
            elif status in ("skip_exists", "dry_run"):
                skipped += 1
            else:
                failed += 1
                print(f"[falha] {info}", file=sys.stderr)

            if use_bar:
                pbar.update(1)

    if use_bar:
        pbar.close()

    print("\nResumo")
    print("-------")
    print(f"Total       : {total}")
    print(f"Convertidos : {converted}")
    print(f"Pulados     : {skipped}")
    print(f"Falhas      : {failed}")
    print(f"Saída       : {root_output}")

if __name__ == "__main__":
    main()