"""Quick test: upload one mask to CTA_002 and print the result."""
import os
import json
import redbrick

p = redbrick.get_project(
    org_id=os.environ["REDBRICK_ORG_ID"],
    project_id=os.environ["REDBRICK_PROJECT_ID"],
    api_key=os.environ["REDBRICK_API_KEY"],
)

task = {
    "taskId": "84ed2294-2d7e-4488-b260-d1c551757fa6",
    "series": [
        {
            "segmentations": r"C:\Users\Samih\Downloads\cta_masks\CTA_003_vessels.nii.gz",
            "segmentMap": {
                "1": {
                    "category": "Vessels"
                }
            }
        }
    ]
}

print("Task payload:")
print(json.dumps(task, indent=2))
print()

result = p.labeling.put_tasks("Label", [task], finalize=True)
print("Result:")
print(json.dumps(result, indent=2, default=str))
