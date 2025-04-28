
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import cv2
from PIL import Image
import torch
import numpy as np
import supervision as sv
from lavis.models import load_model_and_preprocess
import webcolors
import matplotlib.pyplot as plt
import os, shutil
import time
import csv
import skimage
import random

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

#Clase Principal
#directory : Directorio contenedor de imagenes a segmentar
class SegmentadorClasificador():

    def __init__(self, input, segmentador, clasificador, output=None):
        self.directory = input
        self.segmentador = segmentador
        self.clasificador = clasificador
        if output is None: self.output = input
        else : self.output = output

        self.results_dir = self.create_folder()

        self.image_list = []
        self.filedir_list = []

        self.retrieve_images()

    #Ejecucion del programa
    def run(self):

        #Creacion de la estructuras de datos a imprimir y usar
        all_results = {}
        all_results['area'] = []
        all_results['answers'] = []
        all_results['filename'] = []
        all_results['image_name'] = []
        all_results['seg_time'] = 0
        all_results['class_time'] = 0

        #Por cada imagen...
        for i in range(0, len(self.image_list[0])):
            
            #Se segmenta, los resultados se guardan...
            start_t = time.time()
            segmentation_result = self.segmentador.segment(self.image_list[1][i])
            seg_t = time.time() - start_t
            image_to_crop = segmentation_result['image']
            if segmentation_result['image_ratio'] != 1: image_to_crop = self.image_list[1][i]
            segmentation_result = self.parse_seg_results(segmentation_result, image_to_crop, segmentation_result['image_ratio'])

            #Se clasifican los microplasticos individuales por colores...
            start_t = time.time()
            segmentation_result['answers_list'] = self.clasificador.classify(segmentation_result['array_of_plastics_crop'])
            class_t = time.time() - start_t

            #Y los resultados son englosados en una sola estructura de datos a imprimir.
            image_stats = {}
            image_stats['area'] = segmentation_result['area_array']
            image_stats['answers'] = segmentation_result['answers_list']
            image_stats['seg_time'] = seg_t
            image_stats['class_time'] = class_t
            all_results['area'] = all_results['area'] + image_stats['area']
            all_results['answers'] = all_results['answers'] + image_stats['answers']
            all_results['seg_time'] = all_results['seg_time'] + seg_t
            all_results['class_time'] = all_results['class_time'] + class_t
            for k in range(0, len(image_stats['area'])):
                all_results['image_name'].append(self.image_list[0][i])
                all_results['filename'].append("crop" + str(k) + ".jpg")

            #print(segmentation_result['answers_list'])
            file_dir = os.path.join(self.directory, self.image_list[0][i])
            classify_results = self.parse_classify_results(file_dir, segmentation_result['image'], segmentation_result['masks_array'], segmentation_result['bbox_array'], segmentation_result['answers_list'], segmentation_result['area_array'])
            self.to_folder(self.image_list[0][i], segmentation_result['image'], segmentation_result['annotated_image'], segmentation_result['annotated_frame'], 
                           segmentation_result['masks_array'], segmentation_result['array_of_plastics_crop'], classify_results['boxed_image'])
            self.write_text_file(SegmentadorClasificador.crop_filename(self.image_list[0][i]), image_stats)
            
            print("Finalized!")
        self.write_text_file("total", all_results)
        self.write_csv_file("results", all_results)

        print("")
        print("All images successfully analyzed!")

    #Creacion del directorio de salida
    def create_folder(self):
        results_filename = "RESULTS-" + self.segmentador.model_letter
        new_dir = os.path.join(self.output, results_filename)
        if os.path.isdir(new_dir): shutil.rmtree(new_dir)
        os.mkdir(new_dir, mode = 0o777)
        return new_dir

    #Lectura del directorio de entrada
    def retrieve_images(self):
        filename_list = os.listdir(self.directory)
        filename_list_copy = filename_list.copy()
        for file in filename_list_copy:
            if os.path.isfile(self.directory + "/" + file) is False:
                filename_list.remove(file)
            
        if len(filename_list) == 0:
            print("Empty Directory")
            exit(0)

        image_array = []
        for file in filename_list:
            file_path = os.path.join(self.directory, file)
            raw_image = cv2.imread(file_path)
            image_array.append(raw_image)
        self.image_list = [filename_list, image_array]

    #Lectura y englosado de los datos de segmentacion para uso en clasificacion
    def parse_seg_results(self, result_dic, image, image_ratio):

        print("Parsing...")

        masks_array = [
        mask['segmentation']
        for mask
        in sorted(result_dic['masks'], key=lambda x: x['area'], reverse=True)
        ]
        result_dic['masks_array'] = masks_array

        bbox_array = [
            mask['bbox']
            for mask
            in sorted(result_dic['masks'], key=lambda x: x['area'], reverse=True)
        ]
        result_dic['bbox_array'] = bbox_array.copy()

        if image_ratio != 1:
            for i in range(0, len(bbox_array)):
                bbox_array[i] = [int(i * image_ratio) for i in bbox_array[i]]

        area_array = [
            mask['area']
            for mask
            in sorted(result_dic['masks'], key=lambda x: x['area'], reverse=True)
        ]
        result_dic['area_array'] = area_array

        array_of_plastics_crop = []
        for i in range(0, len(masks_array)):
            image_to_mask = np.array(image)
            mask = masks_array[i]

            if image_ratio != 1:
                h, w, l = image_to_mask.shape
                mask = mask * np.uint8(255)
                mask = Image.fromarray(mask)
                mask = mask.resize((w, h), resample=Image.Resampling.NEAREST)
                mask = np.array(mask)
                mask = mask.astype(bool)

            bbox_crop = np.array(bbox_array[i]).astype(int)
            image_to_mask[~mask,:] = [255,255,255]
            #bbox is XYWH, cropping is y:y+h , x:x+w
            cropped_image_final = image_to_mask[bbox_crop[1]:bbox_crop[1]+bbox_crop[3] , bbox_crop[0]:bbox_crop[0]+bbox_crop[2]]
            array_of_plastics_crop.append(cropped_image_final)
        result_dic['array_of_plastics_crop'] = array_of_plastics_crop

        return result_dic

    #Guardado de todos los datos e imagenes en el directorio de salida
    def to_folder(self, filename, og_image, annotated_image, annotated_frame, masks_array, array_of_plastics_crop, boxed_image):
        filename = SegmentadorClasificador.crop_filename(filename)
        file_dir = os.path.join(self.results_dir, filename)
        self.filedir_list.append(file_dir)
        os.mkdir(file_dir, mode = 0o777)
        img0_path = os.path.join(file_dir, "final_image.jpg")
        img1_path = os.path.join(file_dir, "annotated.jpg")
        img2_path = os.path.join(file_dir, "boxed.jpg")
        img3_path = os.path.join(file_dir, "colorboxed.jpg")
        cv2.imwrite(img0_path, og_image)
        cv2.imwrite(img1_path, annotated_image)
        cv2.imwrite(img2_path, annotated_frame)
        cv2.imwrite(img3_path, boxed_image)
        for i in range(0, len(masks_array)):
            crop_path = os.path.join(file_dir, "crop" + str(i) + ".jpg")
            cv2.imwrite(crop_path, array_of_plastics_crop[i])
        
    def parse_classify_results(self, file_dir, image, masks_list, bboxarray_list, answers_list, areaarray_list):

        class_results = {}
        boxed_image = image.copy()
        for j in range(0, len(masks_list)):
            bbox_crop = np.array(bboxarray_list[j]).astype(int)
            string = answers_list[j][0]
            colour = (webcolors.name_to_rgb(answers_list[j][0])[::-1])
            boxed_image = cv2.rectangle(boxed_image, (bbox_crop[0],bbox_crop[1]), (bbox_crop[0]+bbox_crop[2],bbox_crop[1]+bbox_crop[3]), colour, thickness=1)
            boxed_image = cv2.putText(boxed_image, string, (bbox_crop[0],bbox_crop[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)
        #sv.plot_image(image=boxed_image)
        class_results['boxed_image'] = boxed_image

        #path = os.path.join(file_dir, "colorboxed.png")
        #cv2.imwrite(path, boxed_image)

        #labels, counts = np.unique(answers_list, return_counts=True)
        #ticks = range(len(counts))
        #plt.bar(ticks,counts, label = "nocrop")
        #plt.xticks(ticks, labels)
        #plt.savefig(os.path.join(file_dir, "count_histogram.png"))
        #plt.show()

        #area_dict = {}
        #for j in range(len(areaarray_list)):
        #    if answers_list[j][0] in area_dict:
        #        area_dict[answers_list[j][0]] = area_dict[answers_list[j][0]]+areaarray_list[j]
        #    else:
        #        area_dict[answers_list[j][0]] = areaarray_list[j]

        #plt.bar(area_dict.keys(), area_dict.values(), color='b')
        #plt.savefig(os.path.join(file_dir, "area_histogram.png"))
        #plt.show()

        return class_results
    
    #Escritura de fichero csv 
    def write_csv_file(self, filename, image_stats):
        csv_dir = os.path.join(self.results_dir, (filename + ".csv"))
        with open(csv_dir, 'w', newline='') as file:
            writer = csv.writer(file)
            field = ["id", "image_name", "filename", "area", "colour"]

            writer.writerow(field)
            for i in range(0, len(image_stats["area"])):
                new_field = [i, image_stats["image_name"][i], image_stats['filename'][i], image_stats["area"][i], image_stats["answers"][i][0]]
                writer.writerow(new_field)
            
    
    def write_text_file(self, filename, image_stats):
        txt_dir = os.path.join(self.results_dir, filename)
        num_plastics = len(image_stats['area'])
        seg_t = image_stats['seg_time']
        class_t = image_stats['class_time']

        total_area = 0
    
        for area in image_stats['area']:
            total_area = total_area + area

        colour_hist = {}
        for colour in image_stats['answers']:
            colour = colour[0]
            if colour in colour_hist:
                colour_hist[colour] = colour_hist[colour] + 1
            else:
                colour_hist[colour] = 1
        
        area_hist = {}
        for i in range(0, len(image_stats['answers'])):
            colour = image_stats['answers'][i][0]
            area = image_stats['area'][i]
            if colour in colour_hist:
                colour_hist[colour] = colour_hist[colour] + area
            else:
                colour_hist[colour] = area

        f = open(txt_dir + ".txt", "a")
        f.write("Number of microplastics: ")
        f.write(str(num_plastics) + "\n")

        f.write("Total Area: ")
        f.write(str(total_area) + "\n")

        for key, value in colour_hist.items():
            f.write("Amount of " + key + " plastics: ")
            f.write(str(value) + "\n")
        
        for key, value in area_hist.items():
            f.write("Area of " + key + " plastics: ")
            f.write(str(value) + "\n")

        f.write("Segmentation execution time: ")
        f.write(str(seg_t) + "\n")
        f.write("Classification execution time: ")
        f.write(str(class_t) + "\n")
        f.write("Total execution time: ")
        f.write(str(seg_t + class_t) + "\n")

    #Funcion usada multiples veces para la eliminacion del formato de fichero
    @staticmethod
    def crop_filename(filename):
        filename_split = filename.split(".")

        if len(filename_split) > 1:
            filename = ""
            for i in range(0, len(filename_split)-1):
                filename = filename + (filename_split[i] + ".")
            filename = filename[:-1]

        return filename


#Clase segmentadora, detecta y crea mascaras de microplasticos
class Segmentador:

    def __init__(self, model, whitebalancer=None, rescaler=None, overlap_function = None, separate_blobs = None, points_per_side = 32):
        self.model_path = ""
        self.model_name = ""
        self.model_letter = ""
        self.mask_generator = None
        self.whitebalancer = whitebalancer
        self.rescaler = rescaler
        self.overlap_function = overlap_function
        self.separate_blobs = separate_blobs
        self.points_per_side = points_per_side

        self.set_model(model)

        #Se instancia el segmentador, usando CUDA si existe, y si no CPU por defecto
        SAM_DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        sam = sam_model_registry[self.model_name](checkpoint=self.model_path)
        sam.to(device=SAM_DEVICE)
        self.mask_generator = SamAutomaticMaskGenerator(sam, points_per_side = self.points_per_side)

    #Se establece el modelo a usar
    def set_model(self, model):
        if model == 1:
            self.model_path = "models\sam_vit_h_4b8939.pth"
            self.model_name = "default"
            self.model_letter = "H"
        if model == 2:
            self.model_path = "models\sam_vit_l_0b3195.pth"
            self.model_name = "vit_l"
            self.model_letter = "L"
        if model == 3:
            self.model_path = "models\sam_vit_b_01ec64.pth"
            self.model_name = "vit_b"
            self.model_letter = "B"
        if model == 4:
            self.model_path = "models\custom_model.pth"
            self.model_name = "vit_b"
            self.model_letter = "CUSTOM"

    #Se inicia el proceso de segmentacion
    def segment(self, raw_image):
        buffer_image = 0
        image_ratio = 1
        segmentation_result = {}
        #Segmentador.show_image(raw_image)

        #Se hace un balance de blanco sobre la imagen...
        if self.whitebalancer is not None: 
            print("Balancing White...")
            raw_image = self.whitebalancer.balance(raw_image)
            #Segmentador.show_image(raw_image)

        #Se realiza un reescalado...
        if self.rescaler is not None:
            print("Rescaling...")
            buffer_image = raw_image.copy()
            raw_image = self.rescaler.rescale(raw_image)
    
        #Se generan las mascaras...
        print("Detection...")
        array_image = np.array(raw_image)
        masks = self.mask_generator.generate(array_image)
        if len(masks) > 1 : 
            masks.pop(0)

        #Se juntan aquellas mascaras que se superponen...
        if self.overlap_function is not None:
            masks = self.overlap_function.filter_masks(masks)
            for mask in masks:
                converted_mask = mask['segmentation'] * np.uint8(255)
                mask_output = cv2.connectedComponentsWithStats(converted_mask, 8, cv2.CV_32S)
                (numLabels, labels, stats, centroids) = mask_output
                area = stats[1, cv2.CC_STAT_AREA]
                mask['area'] = area

        #Se separan las mascaras que deberian ser distintas...
        if self.separate_blobs is not None:
            masks = self.separate_blobs.filter_masks(masks)

        #Se deduce cuanto se ha decrementado la imagen...
        if self.rescaler is not None:
            h1, w1, l1 = buffer_image.shape
            h2, w2, l2 =  raw_image.shape
            image_ratio = h1 / h2

        #Y se devuelven los datos recopilados.
        print("Printing...")
        mask_annotator = sv.MaskAnnotator(color_lookup = sv.ColorLookup.INDEX)
        detections = sv.Detections.from_sam(masks)
        annotated_image = mask_annotator.annotate(raw_image.copy(), detections)
        bounding_box_annotator = sv.BoundingBoxAnnotator(color_lookup = sv.ColorLookup.INDEX)
        annotated_frame = bounding_box_annotator.annotate(raw_image.copy(), detections)

        #Segmentador.show_image(annotated_image)
        #Segmentador.show_image(annotated_frame)
        segmentation_result['image'] = raw_image
        segmentation_result['masks'] = masks
        segmentation_result['annotated_image'] = annotated_image
        segmentation_result['annotated_frame'] = annotated_frame
        segmentation_result['image_ratio'] = image_ratio
        return segmentation_result
    
    #Funcion para la muestra de imagenes usado durante todo el desarrollo de la segmentacion a modo de prueba
    @staticmethod
    def show_image(image):
        image_pil = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_pil)
        image_pil.show()


#Clase clasificadora, devuelve el color de los microplasticos
class Clasificador:

    def __init__(self, question):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.vis_processors, self.txt_processors = load_model_and_preprocess(name="blip_vqa", model_type="vqav2", is_eval=True, device=self.device)
        self.vis_processors.keys()
        self.txt_processors.keys()
        self.question = question

    #Se inicializa el proceso de clasificacion
    def classify(self, plasticcropsarray_list):
        print("Classifying...")

        #Por cada imagen de microplastico, se recibe una respuesta...
        answers_list = []
        for raw_image in plasticcropsarray_list:
            png_image = Image.fromarray(cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB))
            image = self.vis_processors["eval"](png_image).unsqueeze(0).to(self.device)
            self.question = self.txt_processors["eval"](self.question)
            samples = {"image": image, "text_input": self.question}
            
            answers_list.append(self.model.predict_answers(samples=samples, inference_method="generate"))
        return answers_list
    
#Funcion de superposicion - Junta aquellas mascaras que se superponen.
class OverlapFunction:
    
    def __init__(self, boundary_range):
        self.boundary_range = boundary_range

    def filter_masks(self, masks):
        masks_tuples = []
        for i in range(0, len(masks)-1):
            mask1 = masks[i]['segmentation'].copy()
            kernel = np.ones((3, 3), np.uint8)
            mask1 = np.multiply(mask1, 255)
            mask1 = mask1.astype('uint8')
            mask1 = cv2.dilate(mask1, kernel)
            for j in range(i+1, len(masks)):
                mask2 = masks[j]['segmentation']
                if np.logical_and(mask1, mask2).any():
                    masks_tuples.append([i, j])
        #print(masks_tuples)

        overlap_array = []
        for i in range(0, len(masks_tuples)):
            current_mask = masks_tuples[i]
            if (current_mask):
                j = i+1
                while j < len(masks_tuples):
                #for j in range(i+1, len(masks_tuples)):
                    if np.isin(current_mask, masks_tuples[j]).any():
                        current_mask = current_mask + masks_tuples[j]
                        masks_tuples[j] = []
                        j = i
                    j = j + 1
                overlap_array.append(current_mask)

        for i in range(0, len(overlap_array)):
            overlap_array[i] = list(set(overlap_array[i]))

        for array in overlap_array:
            main_mask = masks[array[0]]
            for i in range(1, len(array)):
                mask_to_merge = masks[array[i]]
                main_mask['segmentation'] = np.logical_or(main_mask['segmentation'], mask_to_merge['segmentation'])
                bbox_x = min(main_mask['bbox'][0], mask_to_merge['bbox'][0])
                bbox_y = min(main_mask['bbox'][1], mask_to_merge['bbox'][1])
                bbox_x2 = max(main_mask['bbox'][0] + main_mask['bbox'][2], mask_to_merge['bbox'][0] + mask_to_merge['bbox'][2])
                bbox_y2 = max(main_mask['bbox'][1] + main_mask['bbox'][3], mask_to_merge['bbox'][1] + mask_to_merge['bbox'][3])
                main_mask['bbox'] = [bbox_x, bbox_y, bbox_x2-bbox_x, bbox_y2-bbox_y]
                masks[array[i]] = []
            masks[array[0]] = main_mask

        masks = [x for x in masks if x != []]
        return masks

#Funcion de separacion - Separa mascaras que muestran mas de un microplastico
class SeparateBlobs:
    def __init__ (self, connectivity = 8):
        self.connectivity = connectivity
        
    def filter_masks(self, masks):
        filtered_masks = []

        for i in range(0, len(masks)):
            mask = masks[i]['segmentation'] * np.uint8(255)

            mask_output = self.separate_blobs(mask)
            (numLabels, labels, stats, centroids) = mask_output

            if numLabels < 3:
                filtered_masks.append(masks[i])
                continue
            
            for k in range(1, numLabels):
                x = stats[k, cv2.CC_STAT_LEFT]
                y = stats[k, cv2.CC_STAT_TOP]
                w = stats[k, cv2.CC_STAT_WIDTH]
                h = stats[k, cv2.CC_STAT_HEIGHT]
                area = stats[k, cv2.CC_STAT_AREA]
            
                filter_mask = (labels!=k)
                new_mask = np.copy(labels)
                new_mask[filter_mask] = new_mask[filter_mask]*0
                new_mask = (new_mask/k).astype(bool)

                mask_dict = {}
                mask_dict['segmentation'] = new_mask
                mask_dict['bbox'] = [x, y, w, h]
                mask_dict['area'] = area

                filtered_masks.append(mask_dict)

        return filtered_masks

    def separate_blobs(self, mask):
        output = cv2.connectedComponentsWithStats(mask, self.connectivity, cv2.CV_32S)
        return output

#Clase de Balance de Blanco
class WhiteBalancer:
    
    def __init__(self, algo, wb_value):
        self.algorythm = algo
        self.wb_value = wb_value

    def balance(self, image):
        if self.algorythm == 1:
            return self.gray_world(image)
        if self.algorythm == 2:
            return self.gray_world_2(image)
        if(self.algorythm) == 3:
            return self.white_patch(image)

    #Funcion de mundo gris - por defecto de cv2
    def gray_world(self, image):
        wb = cv2.xphoto.createGrayworldWB()
        wb.setSaturationThreshold(self.wb_value)
        image = wb.balanceWhite(image, image)
        return image
    
    #Funcion de mundo gris - programada a mano
    def gray_world_2(self, image):
        gray_world_image = skimage.util.img_as_ubyte((image * (image.mean() / image.mean(axis=(0, 1)))).clip(0, 255).astype(int))
        return gray_world_image
    
    #Funcion de Parche Blanco
    def white_patch(self, image, percentile=100):
        white_patch_image = skimage.util.img_as_ubyte((image*1.0 / np.percentile(image,percentile, axis=(0, 1))).clip(0, 1))    
        return white_patch_image
    
#Funcion que reescala imagenes
class Rescaler:

    def __init__(self, algo, model_path=0, model="edsr", model_num=4, downscale_size=1024):
        self.algorithm = algo
        self.model_path = model_path
        self.model = model
        self.model_num = model_num
        self.downscale_size = downscale_size

    def rescale(self, raw_image):
        if self.algorithm == 1:
            return raw_image
        if self.algorithm == 2:
            return self.DNNSS(raw_image)
        if self.algorithm == 3:
            return self.downscale(raw_image)
    
    #Usando DNNSS
    def DNNSS(self, raw_image):
        sr = cv2.dnn_superres.DnnSuperResImpl_create()
        sr.readModel(self.model_path)
        sr.setModel(self.model, self.model_num)
        raw_image = sr.upsample(raw_image)
        return raw_image
    
    #O decrementado tamaño usando cv2
    def downscale(self, raw_image):
        h, w, l = raw_image.shape
        ratio = self.downscale_size / w
        if ratio < 1:
            raw_image = cv2.resize(raw_image, None, fx=ratio, fy=ratio)
        return raw_image
        


##Instanciar WhiteBalancer
#wb = WhiteBalancer(1, 0.99)
#
##Instanciar Rescaler
#path = ""
#model = "edsr"
#model_num = 2
#rescaler = Rescaler(2, path, model, model_num)
#
##Instanciar OverlapFunction
#overlap_function = OverlapFunction(3)
#
##Instanciar SeparateBlobs
#separate_blobs = SeparateBlobs()
#
##Instanciar Segmentador
#seg = Segmentador(3, whitebalancer=wb, rescaler=None, overlap_function=overlap_function, separate_blobs=separate_blobs, points_per_side=64)
#
##Instanciar Clasificador
#clas = Clasificador("What colour is the stone?")
#
##Instanciar SegClas
#folder_path = "C:/Users/Alberto/Desktop/test_area"
#segclas = SegmentadorClasificador(folder_path, seg, clas)
#
##Ejecutar
#segclas.run()