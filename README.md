# SaveIt-FAQ-IA 🤖

API inteligente que responde perguntas sobre o SaveIt usando RAG (Retrieval-Augmented Generation) com Google Gemini. O sistema busca informações em documentos PDF e responde de forma natural e amigável.

## 🌟 Características

- Respostas contextualizadas com base em documentos oficiais
- Tom amigável e natural, entendendo gírias/linguagem informal
- Integração com Google Gemini para processamento de linguagem natural
- Sistema RAG para busca precisa em documentos
- API REST com FastAPI
- Limpeza inteligente de texto para melhor formatação

## 🔧 Tecnologias

- Python 3.x
- FastAPI
- LangChain
- Google Gemini
- FAISS para busca vetorial
- PyPDF Loader

## 📋 Pré-requisitos

1. Python 3.x instalado
2. Chave de API do Google Gemini
3. Arquivo PDF com a documentação do SaveIt
4. Dependências listadas em `requirements.txt`

## 🚀 Instalação

1. Clone o repositório:
```bash
git clone https://github.com/SaveItTeam/SaveIt-FAQ-IA.git
cd SaveIt-FAQ-IA
```

2. Instale as dependências:
```bash
pip install -r requirements.txt
```

3. Configure o arquivo `.env`:
```env
GEMINI_API_KEY=sua_chave_api_aqui
PDF_PATH=caminho/para/seu/documento.pdf
```

## 💻 Como Usar

1. Inicie o servidor:
```bash
uvicorn app:app --reload
```

2. Faça uma requisição para a API:
```bash
curl -X POST "http://localhost:8000/faq" \
     -H "Content-Type: application/json" \
     -d '{"question": "O que é o SaveIt?", "session_id": "WEB"}'
```

## 📚 Estrutura do Projeto

```
SaveIt-FAQ-IA/
├── app.py           # API principal e configuração do LLM
├── faq_tool.py      # Funções de processamento do PDF e contexto
├── requirements.txt  # Dependências do projeto
└── .env             # Variáveis de ambiente (não versionado)
```

## 🛠️ Funções Principais

- `get_faq_context()`: Busca informações relevantes no PDF
- `clean_text()`: Remove formatações e padroniza o texto
- `answer_question()`: Processa perguntas e gera respostas

## 📝 Exemplo de Uso

```python
from app import answer_question

# Fazendo uma pergunta
resposta = answer_question(
    "Como faço para gerenciar produtos?",
    session_id="WEB"
)
print(resposta)
```

## 👥 Contribuindo

1. Faça um fork do projeto
2. Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add: nova feature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

## 📄 Licença

Este projeto está sob a licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.

## ✨ Agradecimentos

- Time SaveIt pelo suporte e documentação
- Contribuidores do projeto
- Comunidade LangChain e FastAPI
