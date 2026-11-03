"""Reproducible 12x9 squares / 30 mm simulated end-to-end test. No hardware."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
import cv2
import numpy as np
if __package__:
    from . import calibrate as c
    from . import table_calibration as t
    from .synthetic_demo import render, make_T, save_csv
else:  # preserve the original standalone demo
    import calibrate as c
    import table_calibration as t
    from synthetic_demo import render, make_T, save_csv


def run(root):
    root = c.output_dir(root).resolve()
    (root / "images").mkdir()
    rng = np.random.default_rng(20261019)
    K = np.array([[1000., 0, 640], [0, 990., 480], [0, 0, 1.]])
    size = (1280, 960)
    manifests = []
    for i in range(24):
        R = cv2.Rodrigues(rng.uniform([-.6,-.6,-.4],[.6,.6,.4]))[0]
        centre = rng.uniform([-.12,-.08,.65],[.12,.08,.95])
        C = c.transform(R, centre - R @ np.array([.15,.105,0]))
        name = f"images/view_{i:02d}.png"
        c.save_image(root / name, render(C,K,cols=11,rows=8,s=.03,size=size))
        manifests.append({"image":name,"split":"train" if i<18 else "val"})
    save_csv(root/"intrinsics.csv",["image","split"],manifests)
    C = make_T([.16,-.12,.07],[-.14,-.10,.78])
    Rb = cv2.Rodrigues(np.array([0.,0.,.3]))[0] @ np.diag([1.,-1.,-1.])
    B = c.transform(Rb,[.20,.15,.03])
    X = B @ c.inverse(C)
    c.save_image(root/"reference.png",render(C,K,cols=11,rows=8,s=.03,size=size))
    _,_,corners=c.detect(root/"reference.png",11,8)
    origin=cv2.projectPoints(np.zeros((1,3)),cv2.Rodrigues(C[:3,:3])[0],C[:3,3],K,np.zeros(5))[0].reshape(2)
    order="normal" if np.linalg.norm(corners[0,0]-origin)<np.linalg.norm(corners[-1,0]-origin) else "reverse"
    scripts=Path(__file__).resolve().parent
    logs=[]
    def command(script,*args,expected=0):
        cmd=[sys.executable,str(scripts/script),*map(str,args)]
        result=subprocess.run(cmd,capture_output=True,text=True)
        logs.append({"command":cmd[2:],"script":script,"exit_code":result.returncode,
                     "stdout":result.stdout,"stderr":result.stderr})
        print(result.stdout.strip() or result.stderr.strip(),flush=True)
        if result.returncode!=expected:
            raise AssertionError(f"{script}: expected exit {expected}, got {result.returncode}: {result.stderr}")
    command("calibrate.py","intrinsics","--manifest",root/"intrinsics.csv","--cols",11,"--rows",8,
            "--square-m",.03,"--camera-id","SIMULATED_CAMERA","--max-rms-px",1,"--out",root/"intrinsics")
    camera=root/"intrinsics/intrinsics.json"
    command("table_calibration.py","detect","--image",root/"reference.png","--corner-order",order,"--out",root/"board")
    obj=c.board_points(**t.BOARD)
    teach=[]
    for split,ids in [("train",[0,5,10,33,43,77,82,87]),("val",[14,25,49,69])]:
        for i in ids:
            xyz=t.apply(B,obj[i])+rng.normal(0,.00015,3)
            teach.append(dict(zip(["corner_id","split","x","y","z"],[i,split,*(xyz*1000)])))
    save_csv(root/"teach.csv",list(teach[0]),teach)
    shared=["fit","--intrinsics",camera,"--observation",root/"board/observation.json",
            "--unit","mm","--robot-id","SIM_NOVA2_LEFT","--base-frame","SIM_BASE","--tcp-name","SIM_TIP",
            "--max-rms-px",1,"--max-error-mm",2]
    command("table_calibration.py",*shared,"--points",root/"teach.csv","--out",root/"extrinsics")
    extrinsics=root/"extrinsics/extrinsics.json"
    fitted=c.load_json(extrinsics)
    err=c.residual(np.array(fitted["T_base_camera"]),X)
    if err["translation_mm"]>8 or err["rotation_deg"]>1:
        raise AssertionError(f"Synthetic transform recovery failed: {err}")
    checks=[]
    for pb in ([.04,.05,0],[.24,.06,0],[.22,.18,0],[.10,.14,0]):
        uv=cv2.projectPoints(np.array([pb]),cv2.Rodrigues(C[:3,:3])[0],C[:3,3],K,np.zeros(5))[0].reshape(2)
        reference=t.apply(B,pb)
        checks.append(dict(zip(["u_px","v_px","board_z_m","reference_x_m","reference_y_m","reference_z_m"],[*uv,0,*reference])))
    save_csv(root/"independent.csv",list(checks[0]),checks)
    command("table_calibration.py","verify","--intrinsics",camera,"--extrinsics",extrinsics,
            "--robot-id","SIM_NOVA2_LEFT","--csv",root/"independent.csv","--max-error-mm",2,"--out",root/"validation")
    pixel=[checks[0]["u_px"],checks[0]["v_px"]]
    command("table_calibration.py","point","--intrinsics",camera,"--extrinsics",extrinsics,
            "--robot-id","SIM_NOVA2_LEFT","--pixel",*pixel,"--board-z-m",0)
    # Prove held-out measurements cannot change the fitted transform.
    bad=[dict(r) for r in teach]
    for row in bad:
        if row["split"]=="val":
            row["x"]+=50
    save_csv(root/"bad_val.csv",list(bad[0]),bad)
    command("table_calibration.py",*shared,"--points",root/"bad_val.csv","--out",root/"rejected",expected=2)
    np.testing.assert_allclose(fitted["T_base_camera"],c.load_json(root/"rejected/extrinsics.json")["T_base_camera"],atol=1e-12)
    command("table_calibration.py","point","--intrinsics",camera,"--extrinsics",extrinsics,
            "--robot-id","WRONG_ARM","--pixel",*pixel,"--board-z-m",0,expected=1)
    # Correct magnitudes but wrong declared units must fail metric consistency.
    wrong=list(shared)
    wrong[wrong.index("mm")]="m"
    command("table_calibration.py",*wrong,"--points",root/"teach.csv","--out",root/"rejected_units",expected=2)
    summary={"synthetic_only":True,"real_hardware_tested":False,"board":t.BOARD,
             "image_views":24,"train_views":18,"validation_views":6,
             "train_intrinsics_rms_px":c.load_json(camera)["train_rms_px"],
             "T_base_camera_error":err,"independent_plane_points":c.load_json(root/"validation/validation.json"),
             "negative_checks":{"bad_validation_rejected":True,"heldout_does_not_change_fit":True,
                                "wrong_robot_rejected":True,"wrong_units_rejected":True},"commands":logs}
    # Every JSON produced by this demonstration is explicitly simulated.
    for path in root.rglob("*.json"):
        data=c.load_json(path)
        data["synthetic_only"]=True
        c.write_json(path,data)
    # Update internal hashes after marking simulated camera/observation files.
    for path in root.rglob("extrinsics.json"):
        data=c.load_json(path)
        data["intrinsics_sha256"]=t.digest(camera)
        data["observation_sha256"]=t.digest(root/"board/observation.json")
        c.write_json(path,data)
    c.write_json(root/"synthetic_table_report.json",summary)
    print(json.dumps({k:v for k,v in summary.items() if k!="commands"},indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",required=True)
    run(parser.parse_args().out)
