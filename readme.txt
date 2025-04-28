Todos los enlances se hayan al final del documento de instalacion.

Instalar Python, version mínima 3.9 requerida.
Instalar Pytorch y Torchvision.

Usando el entorno python de su conveniencia, por ejemplo, el CMD de Windows, 
instalar las siguientes librerias:
	-segment-anything - Desde Enlace
	-salesforce-lavis - Desde Enlace
	-supervision - "pip install supervision"
	-webcolors - "pip install webcolors"
	-skimage - "pip install skimage"
	-PyQt5 - Desde enlace QtDesigner
	
Descargar los siguientes modelos que se encuentran 
en el git de segment-anything:
	-vit_h (REQUERIDO)
	-vit_l (OPCIONAL)
	-vit_b (OPCIONAL)

Descargar la aplicacion y descomprimir.

Colocar modelos en la carpeta "models"

Para ejecutar el programa en lina de comando por CMD de Windows, 
usar "python segment_command.py"
Para ejecutar el programa con interfaz grafica por CMD de Windows, 
usar el comando "python segment_ui.py"

EJECUCIÓN SIMPLE:
EJECUTAR "execute.bat", Y EL SEGMENTADOR CON UI EJECUTARÁ AUTOMÁTICAMENTE.






EJECUCIÓN:
segment_command.py:
Programa de ejecucion por terminal.
Ejecutar usando comando "python" por terminal, o interfaz de desarrollo 
preferida.
Usar opcion "-h" para ver menu de ayuda.

Ejemplo (Ejecutando por CMD de Windows):
Primero, acceder a la carpeta contenedora por linea de comando.
"python segment_command.py -h" - Menu de ayuda del programa

"python segment_command.py "C:\Users\Ejemplo\Desktop\Carpeta_De_Microplasticos"
Ejecutara el programa usando la carpeta "Carpeta de microplasticos"

EJECUCION:
segment_ui.py:
Programa de interfaz gráfica.
Primero, acceder a la carpeta contenedora por linea de comando.
Ejecutar usando comando "python" por terminal, o interfaz de desarrollo 
preferida.
Tener en cuenta que desde el CMD de Windows tenemos que estar
dentro de la carpeta contenedora del programa.
Ejemplo: "python segment_ui.py"





ENLACES:
Python: https://www.python.org/downloads/
Pytorch y Torchvision: https://pytorch.org/get-started/locally/
Segment Anything: https://github.com/facebookresearch/segment-anything
Modelos:
	vit_h: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
	vit_l: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
	vit_b: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
LAVIS: https://github.com/salesforce/LAVIS
Supervision: https://github.com/roboflow/supervision
QtDesigner: https://build-system.fman.io/qt-designer-download

Segmentador de Microplasticos: 
    https://github.com/AlbertoFidalgo/Microplastic-Segmenter