import numpy as np
import os
from fastapi import FastAPI, File, UploadFile, HTTPException, APIRouter
from fastapi.responses import JSONResponse, StreamingResponse
import io
import tempfile
from src.model.prediction import mask_output
from src.configuration.config import weights
from src.model.load_model import model
from pathlib import Path
import shutil
import cv2
import highdicom as hd
import pydicom
from pydicom.sr.codedict import codes
import tempfile
import os
from src.utils.dicom_utils import dicom_to_array
from src.model.prediction import mask_output
import traceback
from src.configuration.config import DICOM_TEMP_PATH,JPEG_TEMP_PATH
from fastapi import Response
from src.utils import CommonUtils
from src.configuration.config import OutputDir
from zipfile import ZipFile
import glob


router = APIRouter()

@router.post("/segment_dicom/")
async def segment_dicom(file: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".dcm") as temp_dicom:
            DicomPath = temp_dicom.name
            temp_dicom.write(await file.read())


        output_filename = os.path.basename(DicomPath).replace(".dcm", ".jpg")
        output_path = os.path.join(JPEG_TEMP_PATH, output_filename)
        
        pixel_array=dicom_to_array(DicomPath,output_path,format="jpg")
        mask=mask_output(pixel_array,model)
        mask = (mask > 0).astype(np.uint8)
        print(f"Mask shape: {mask.shape}")
        print(mask.dtype)  # Should be uint8 or int

        print(f"Unique values in mask: {np.unique(mask)}")
        if mask is None:
            raise HTTPException(status_code=500, detail="Error generating segmentation mask")

        source_images = []
                
        source_images.append(pydicom.dcmread(DicomPath))

        description = hd.seg.SegmentDescription(
            segment_number=1,
            segment_label='Fracture',
            segmented_property_category=codes.SCT.Blood,
            segmented_property_type=codes.SCT.Blood,
            algorithm_type=hd.seg.SegmentAlgorithmTypeValues.MANUAL,
        )

        seg_obj = hd.seg.Segmentation(
                        source_images=source_images,
                        pixel_array=mask,
                        segmentation_type=hd.seg.SegmentationTypeValues.BINARY,
                        segment_descriptions=[description],
                        series_instance_uid=hd.UID(),
                        series_number=1,
                        sop_instance_uid=hd.UID(),
                        instance_number=1,
                        manufacturer='Radpretation ai',
                        manufacturer_model_name='Brain structure Segmentation Algorithm',
                        software_versions='0.0.1',
                        device_serial_number='1234567890'
                    )
        seg_obj_path  = os.path.join(OutputDir,'AI_SEG.dcm')
        seg_obj.save_as(seg_obj_path)

        temp_dir_return = tempfile.mkdtemp(dir=DICOM_TEMP_PATH)
        shutil.move(seg_obj_path, temp_dir_return)
        data = {
            "file_id": os.path.basename(temp_dir_return)
        }

        return JSONResponse(content=data)

        # return StreamingResponse(
        #     open(temp_file_path, "rb"),
        #     media_type="application/dicom",
        #     headers={"Content-Disposition": f'attachment; filename="segmented.dcm"'},
        # )
    except Exception as e:
        error_trace = traceback.format_exc()  # Get detailed error traceback
        print("Error Traceback:", error_trace)  # Log it for debugging
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}\n\nTraceback:\n{error_trace}")




@router.post("/segment_dicomv2/")
async def segment_dicom(file: UploadFile = File(...)):
    temp_dir = tempfile.mkdtemp(dir=DICOM_TEMP_PATH)
    try:
        temp_file = os.path.join(temp_dir, file.filename)
        with open(temp_file, "wb") as out_file:
            out_file.write(await file.read())

        with ZipFile(temp_file, 'r') as zip_ref:
            root_dir = zip_ref.namelist()[0].split('/')[0]  # Find the root dir in the zip
            zip_ref.extractall(temp_dir)

        root_dir_path = os.path.join(temp_dir, root_dir)

        file_paths = glob.glob(root_dir_path + '/**/*.dcm', recursive=True)
        DicomPath = file_paths[0]






        
        # with tempfile.NamedTemporaryFile(delete=False, suffix=".dcm") as temp_dicom:
        #     DicomPath = temp_dicom.name
        #     temp_dicom.write(await file.read())


        output_filename = os.path.basename(DicomPath).replace(".dcm", ".jpg")
        output_path = os.path.join(JPEG_TEMP_PATH, output_filename)
        
        pixel_array=dicom_to_array(DicomPath,output_path,format="jpg")
        mask=mask_output(pixel_array,model)
        mask = (mask > 0).astype(np.uint8)
        print(f"Mask shape: {mask.shape}")
        print(mask.dtype)  # Should be uint8 or int

        print(f"Unique values in mask: {np.unique(mask)}")
        if mask is None:
            raise HTTPException(status_code=500, detail="Error generating segmentation mask")

        source_images = []
                
        source_images.append(pydicom.dcmread(DicomPath))

        description = hd.seg.SegmentDescription(
            segment_number=1,
            segment_label='Fracture',
            segmented_property_category=codes.SCT.Blood,
            segmented_property_type=codes.SCT.Blood,
            algorithm_type=hd.seg.SegmentAlgorithmTypeValues.MANUAL,
        )

        seg_obj = hd.seg.Segmentation(
                        source_images=source_images,
                        pixel_array=mask,
                        segmentation_type=hd.seg.SegmentationTypeValues.BINARY,
                        segment_descriptions=[description],
                        series_instance_uid=hd.UID(),
                        series_number=1,
                        sop_instance_uid=hd.UID(),
                        instance_number=1,
                        manufacturer='Radpretation ai',
                        manufacturer_model_name='Brain structure Segmentation Algorithm',
                        software_versions='0.0.1',
                        device_serial_number='1234567890'
                    )
        seg_obj_path  = os.path.join(OutputDir,'AI_SEG.dcm')
        seg_obj.save_as(seg_obj_path)

        temp_dir_return = tempfile.mkdtemp(dir=DICOM_TEMP_PATH)
        shutil.move(seg_obj_path, temp_dir_return)
        data = {
            "file_id": os.path.basename(temp_dir_return)
        }

        return JSONResponse(content=data)

        # return StreamingResponse(
        #     open(temp_file_path, "rb"),
        #     media_type="application/dicom",
        #     headers={"Content-Disposition": f'attachment; filename="segmented.dcm"'},
        # )
    except Exception as e:
        error_trace = traceback.format_exc()  # Get detailed error traceback
        print("Error Traceback:", error_trace)  # Log it for debugging
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}\n\nTraceback:\n{error_trace}")
    


@router.get("/seg/{temp_dir}")
async def get_seg_object(temp_dir: str):
    try:
        path = os.path.join(DICOM_TEMP_PATH,temp_dir, "AI_SEG.dcm")
        dir_path = os.path.join(DICOM_TEMP_PATH,temp_dir)
        if os.path.exists(path):
            file_content = ""
            with open(path, 'rb') as f:
                file_content =f.read()
            return Response(content=file_content, media_type="application/octet-stream")
        else:
            raise HTTPException(status_code=404, detail="File not found")
    except Exception as e:
        raise e
    finally:
        # Whether we had an error or not, it's important to clean up the temp directory
        os.remove(path)
        CommonUtils.delete_if_empty(dir_path)


