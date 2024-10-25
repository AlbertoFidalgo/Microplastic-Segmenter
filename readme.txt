Se recomienda leer el documento completo antes de la ejecucion del programa.

INSTALACION:
Usando el entorno python de su conveniencia, instalar las siguientes librerias:
	-segment-anything
	-salesforce-lavis
	-supervision
	-webcolors
	-skimage
	-PyQt5
	
Descargar los siguientes modelos que se encuentran en el git de segment-anything:
	-vit_h (REQUERIDO)
	-vit_l (OPCIONAL)
	-vit_b (OPCIONAL)

Colocar modelos en la carpeta "models"

EJECUCION:
segment_command.py:
Programa de ejecucion por terminal.
Ejecutar usando comando "python" por terminal, o interfaz de desarrollo preferida.
Usar opcion "-h" para ver funcionamiento.

segment_ui.py:
Programa de ejecucion por interfaz grafica.
Ejecutar usando comando "python" por terminal, o interfaz de desarrollo preferida.
	Input Directory: Carpeta donde se encuentran todas las imagenes a escanear. REQUERIDO
	Output Directory: Carpeta donde se creara la carpeta de salida de datos. REQUERIDO
	Points Per Side: Cantidad de puntos de segmentacion de segment-anything, se recomienda y usa 64 por defecto. REQUERIDO
	Run: Inicializa la ejecucion del programa.
	
FUNCIONAMIENTO:
El programa esta diseñado para la busqueda y clasificacion de microplásticos en una imagen de fondo blanco,
usando las librerias de segment-anything y LAVIS.
Localizará en la imagen los microsplásticos, y clasificará por color.

ENLACES:
Segment Anything: https://github.com/facebookresearch/segment-anything
	modelo vit_h: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
	modelo vit_l: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
	modelo vit_b: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
LAVIS: https://github.com/salesforce/LAVIS
Supervision: https://github.com/roboflow/supervision
QtDesigner: https://build-system.fman.io/qt-designer-download

	