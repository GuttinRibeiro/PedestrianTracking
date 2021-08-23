# PedestrianTracking
Este repositório contém exemplos de códigos desenvolvidos para rastrear pedestres em imagens a partir das janelas de detecção fornecidas por um algoritmo externo. No momento, foi implementado um filtro de Kalman para a tarefa, considerando como estado a posição (x, y), a aultura e a largura da bouning box, bem como as velocidades de variação dessas grandezas. Tarefa baseada no MOT Challenge.


# Resultados (MOT 2015)

https://user-images.githubusercontent.com/31515305/130448504-2b949630-def4-4a12-b478-653cd91c3ecb.mp4

# Tarefas
1) Implementar filtro de Kalman (ok);
2) Saída na forma de vídeo (ok);
3) Saída na forma de arquivo para o MOT Challenge (ok);
4) Adicionar argumentos de linha de comando (TODO);
5) Avaliar resultados com as métricas do desafio a partir do arquivo de saída (TODO);
6) Implementar filtro de partículas (TODO);
7) Utilizar o OpenPose para fazer a detecção dos pedestres (TODO);
8) Implementar uma CNN para detectar os pedestres (TODO).
