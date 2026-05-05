# Python Virtual Environment (venv) + requirements.txt + VS Code Setup  
# Ambiente Virtual Python (venv) + requirements.txt + Instalação no VS Code

---

## English

This document explains how to:

- activate a Python virtual environment (`venv`)
- generate a `requirements.txt`
- install dependencies from `requirements.txt`
- configure and use the project in Visual Studio Code

---

### 1) Generate `requirements.txt`

After activating your virtual environment and installing your libraries, run:

```bash
pip freeze > requirements.txt
```

This command creates a file with all installed packages and their versions.

Example:

```bash
pip freeze > requirements.txt
```

---

### 2) Create a virtual environment

Inside your project folder, run:

```bash
python -m venv .venv
```

You can also use:

```bash
py -m venv .venv
```

---

### 3) Activate the virtual environment

#### Windows (Command Prompt)

```bash
.venv\Scripts\activate
```

#### Windows (PowerShell)

```powershell
.venv\Scripts\Activate.ps1
```

#### Linux / macOS

```bash
source .venv/bin/activate
```

After activation, your terminal usually shows the environment name, for example:

```bash
(.venv)
```

---

### 4) Install packages inside the virtual environment

Example:

```bash
pip install pandas numpy scikit-learn matplotlib
```

If you already have a `requirements.txt`, run:

```bash
pip install -r requirements.txt
```

---

### 5) Recreate the environment from `requirements.txt`

If another person downloads your project, they can run:

```bash
python -m venv .venv
```

Activate the environment, then install dependencies:

```bash
pip install -r requirements.txt
```

---

### 6) Recommended commands before exporting dependencies

It is a good idea to upgrade `pip` first:

```bash
python -m pip install --upgrade pip
```

Then export packages:

```bash
pip freeze > requirements.txt
```

---

### 7) How to use this project in Visual Studio Code

#### Install VS Code
Download and install Visual Studio Code from the official website.

#### Install the Python extension
In VS Code:

1. Open **Extensions**
2. Search for **Python**
3. Install the extension published by Microsoft

#### Open the project folder
In VS Code:

- click **File > Open Folder**
- select your project folder

#### Select the Python interpreter
Press:

```bash
Ctrl + Shift + P
```

Then search for:

```text
Python: Select Interpreter
```

Choose the interpreter from your virtual environment, usually something like:

```text
.venv\Scripts\python.exe
```

---

### 8) Open the terminal in VS Code

In VS Code:

- click **Terminal > New Terminal**

Then activate the environment.

#### Windows CMD

```bash
.venv\Scripts\activate
```

#### Windows PowerShell

```powershell
.venv\Scripts\Activate.ps1
```

If PowerShell blocks script execution, run:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then try again.

---

### 9) Useful workflow

A simple workflow is:

```bash
python -m venv .venv
```

```bash
.venv\Scripts\activate
```

```bash
python -m pip install --upgrade pip
```

```bash
pip install -r requirements.txt
```

If you are creating the project from scratch:

```bash
python -m venv .venv
```

```bash
.venv\Scripts\activate
```

```bash
pip install pandas numpy scikit-learn matplotlib
```

```bash
pip freeze > requirements.txt
```

---


```

---

## Português (BR)

Este documento explica como:

- ativar um ambiente virtual Python (`venv`)
- gerar um arquivo `requirements.txt`
- instalar dependências a partir do `requirements.txt`
- configurar e usar o projeto no Visual Studio Code

---

### 1) Gerar o `requirements.txt`

Depois de ativar seu ambiente virtual e instalar as bibliotecas, execute:

```bash
pip freeze > requirements.txt
```

Esse comando cria um arquivo com todos os pacotes instalados e suas versões.

Exemplo:

```bash
pip freeze > requirements.txt
```

---

### 2) Criar um ambiente virtual

Dentro da pasta do projeto, execute:

```bash
python -m venv .venv
```

Ou:

```bash
py -m venv .venv
```

---

### 3) Ativar o ambiente virtual

#### Windows (Prompt de Comando)

```bash
.venv\Scripts\activate
```

#### Windows (PowerShell)

```powershell
.venv\Scripts\Activate.ps1
```

#### Linux / macOS

```bash
source .venv/bin/activate
```

Após ativar, normalmente o terminal mostra algo assim:

```bash
(.venv)
```

---

### 4) Instalar pacotes dentro do ambiente virtual

Exemplo:

```bash
pip install pandas numpy scikit-learn matplotlib
```

Se você já possui um `requirements.txt`, execute:

```bash
pip install -r requirements.txt
```

---

### 5) Recriar o ambiente a partir do `requirements.txt`

Se outra pessoa baixar seu projeto, ela pode executar:

```bash
python -m venv .venv
```

Ativar o ambiente e depois instalar as dependências:

```bash
pip install -r requirements.txt
```

---

### 6) Comandos recomendados antes de exportar dependências

É uma boa prática atualizar o `pip` primeiro:

```bash
python -m pip install --upgrade pip
```

Depois exportar os pacotes:

```bash
pip freeze > requirements.txt
```

---

### 7) Como usar este projeto no Visual Studio Code

#### Instalar o VS Code
Baixe e instale o Visual Studio Code no site oficial.

#### Instalar a extensão Python
No VS Code:

1. Abra **Extensions**
2. Procure por **Python**
3. Instale a extensão publicada pela Microsoft

#### Abrir a pasta do projeto
No VS Code:

- clique em **File > Open Folder**
- selecione a pasta do seu projeto

#### Selecionar o interpretador Python
Pressione:

```bash
Ctrl + Shift + P
```

Depois procure por:

```text
Python: Select Interpreter
```

Escolha o interpretador da sua `venv`, normalmente algo como:

```text
.venv\Scripts\python.exe
```

---

### 8) Abrir o terminal no VS Code

No VS Code:

- clique em **Terminal > New Terminal**

Depois ative o ambiente.

#### Windows CMD

```bash
.venv\Scripts\activate
```

#### Windows PowerShell

```powershell
.venv\Scripts\Activate.ps1
```

Se o PowerShell bloquear a execução de scripts, execute:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Depois tente novamente.

---

### 9) Fluxo de trabalho útil

Um fluxo simples é:

```bash
python -m venv .venv
```

```bash
.venv\Scripts\activate
```

```bash
python -m pip install --upgrade pip
```

```bash
pip install -r requirements.txt
```

Se você estiver criando o projeto do zero:

```bash
python -m venv .venv
```

```bash
.venv\Scripts\activate
```

```bash
pip install pandas numpy scikit-learn matplotlib
```

```bash
pip freeze > requirements.txt
```

---



