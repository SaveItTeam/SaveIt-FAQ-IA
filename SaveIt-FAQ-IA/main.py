import argparse
import uvicorn
from app import answer_question

def main():
    parser = argparse.ArgumentParser(description="SaveIt FAQ AI Assistant. Run as API server or CLI.")
    parser.add_argument("--serve", action="store_true", help="Rodar a API HTTP com Uvicorn.")
    args = parser.parse_args()

    if args.serve:
        # Inicia o servidor Uvicorn
        print("Iniciando servidor Uvicorn...")
        uvicorn.run("app:app", host="0.0.0.0", port=5000, reload=True)
    else:
        # Modo CLI (Linha de Comando)
        print("FAQ assistant (digite 'sair' para encerrar)")
        while True:
            q = input("Pergunta: ").strip()
            if q.lower() in ("sair", "exit", "quit"):
                break
            try:
                # Chama a função de resposta que está no app.py
                print(answer_question(q))
            except Exception as e:
                print("Erro:", e)

if __name__ == "__main__":
    main()
