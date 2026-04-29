@echo off
echo Criando estrutura de pastas...

mkdir data
mkdir data\processed
mkdir data\raw
mkdir data\features
mkdir data\predictions
mkdir data\exports

mkdir docs
mkdir docs\img
mkdir docs\pdf

mkdir models
mkdir models\baseline
mkdir models\metrics
mkdir models\tuned

mkdir notebooks

mkdir reports

mkdir src
mkdir src\config
mkdir src\dataProcessing
mkdir src\loaders
mkdir src\models
mkdir src\pipelines
mkdir src\repository
mkdir src\visualization

echo Estrutura criada com sucesso.
pause