import uuid
from pydantic import BaseModel

from fastapi import FastAPI, File, UploadFile
import base64
import os, glob
import json

from api.util import modify_pose_json
from inference import get_default_args, infer_model, infer_model_pose, get_default_args_pose


class Item(BaseModel):
    content: str


app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/ai")
async def get_inference_from_image(item: Item):
    UPLOAD_DIR = "./api/images"

    uid = str(uuid.uuid4())
    image_name = f'{uid}.jpg'
    with open(os.path.join(UPLOAD_DIR, image_name), 'wb') as fp:
        fp.write(base64.b64decode(item.content))

    infer_model(get_default_args(uid))
    infer_model_pose(get_default_args_pose(uid))

    with open(f"./api/outputs/{uid}_1.json", "r") as f:
        output_object = json.load(f)
    with open(f"./api/outputs/{uid}_2.json", "r") as f:
        output_pose = json.load(f)
        modified_pose = modify_pose_json(output_pose, "category_id")
    output = output_object + modified_pose

    os.remove(f"./api/outputs/{uid}_1.json")
    os.remove(f"./api/outputs/{uid}_2.json")
    os.remove(f"./api/outputs/{uid}.jpg")
    os.remove(f"./api/images/{uid}.jpg")

    return output


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=80)
