import json
import re
import subprocess


def create_instance(project_id, zone, instance_name):
    try:
        subprocess.check_call(
            [
                "gcloud",
                "compute",
                "instances",
                "create",
                instance_name,
                "--project",
                project_id,
                "--zone",
                zone,
                "--machine-type",
                "n1-highcpu-16",
                "--accelerator",
                "count=1,type=nvidia-tesla-t4",
                "--network-interface",
                "network-tier=PREMIUM,stack-type=IPV4_ONLY,subnet=default",
                "--maintenance-policy",
                "TERMINATE",
                "--provisioning-model",
                "STANDARD",
                "--no-service-account",
                "--no-scopes",
                "--tags",
                "http-server,https-server",
                "--create-disk",
                "auto-delete=yes,boot=yes,device-name=instance-3,image=projects/openmind-vis/global/images/gcp-main,mode=rw,size=60,type=projects/openmind-vis/zones/us-central1-a/diskTypes/pd-balanced",
                "--labels",
                "goog-ec-src=vm_add-gcloud",
                "--reservation-affinity",
                "any",
            ]
        )
        print(f"Instance {instance_name} created in {zone}")
        return True
    except subprocess.CalledProcessError:
        print(f"Failed to create instance in {zone}")
        return False


def get_zones(project_id, region):
    zones_raw = subprocess.check_output(
        [
            "gcloud",
            "compute",
            "zones",
            "list",
            "--project",
            project_id,
            "--filter",
            f"region:{region}",
            "--format",
            "json",
        ]
    )
    zones = json.loads(zones_raw)
    return [zone["name"] for zone in zones]


def get_regions():
    regions_raw = subprocess.check_output(
        ["gcloud", "compute", "regions", "list", "--format", "json"]
    )
    regions = json.loads(regions_raw)
    return [region["name"] for region in regions]


def get_next_instance_name(project_id):
    instances_raw = subprocess.check_output(
        [
            "gcloud",
            "compute",
            "instances",
            "list",
            "--project",
            project_id,
            "--format",
            "json",
        ]
    )
    instances = json.loads(instances_raw)
    instance_numbers = [
        int(re.search(r"instance-(\d+)", instance["name"]).group(1))
        for instance in instances
        if re.match(r"instance-\d+", instance["name"])
    ]
    if instance_numbers:
        max_instance_number = max(instance_numbers)
        return f"instance-{max_instance_number + 1}"
    else:
        return "instance-1"


def main():
    project_id = "openmind-vis"  # Replace with your project ID
    regions = get_regions()  # Replace or expand as needed
    instance_name = get_next_instance_name(
        project_id
    )  # Replace with your instance name

    for region in regions:
        zones = get_zones(project_id, region)
        for zone in zones:
            success = create_instance(project_id, zone, instance_name)
            if success:
                return


if __name__ == "__main__":
    main()
