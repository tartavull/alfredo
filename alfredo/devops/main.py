import json
import operator
import os
import re
import subprocess
import time

from geopy.distance import geodesic
from geopy.geocoders import Nominatim
from joblib import Memory
from tqdm import tqdm

# Create a folder for cache
cachedir = "/tmp/alfredo-cache"
if not os.path.exists(cachedir):
    os.makedirs(cachedir)
memory = Memory(cachedir, verbose=0)


def create_instance(project_id, zone, instance_name, machine_type, image_name):
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
                machine_type,
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
                f"auto-delete=yes,boot=yes,device-name=instance-3,image=projects/openmind-vis/global/images/{image_name},mode=rw,size=60,type=projects/openmind-vis/zones/us-central1-a/diskTypes/pd-balanced",
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


region_locations = {
    "asia-east1": "Changhua County, Taiwan",
    "asia-east2": "Hong Kong",
    "asia-northeast1": "Tokyo, Japan",
    "asia-northeast2": "Osaka, Japan",
    "asia-northeast3": "Seoul, South Korea",
    "asia-south1": "Mumbai, India",
    "asia-south2": "Delhi, India",
    "asia-southeast1": "Jurong West, Singapore",
    "asia-southeast2": "Jakarta, Indonesia",
    "australia-southeast1": "Sydney, Australia",
    "australia-southeast2": "Melbourne, Australia",
    "europe-central2": "Warsaw, Poland",
    "europe-north1": "Hamina, Finland",
    "europe-southwest1": "Lisbon, Portugal",
    "europe-west1": "St. Ghislain, Belgium",
    "europe-west2": "London, UK",
    "europe-west3": "Frankfurt, Germany",
    "europe-west4": "Eemshaven, Netherlands",
    "europe-west6": "Zurich, Switzerland",
    "europe-west8": "Doha, Qatar",
    "europe-west9": "Milan, Italy",
    "me-central1": "Manama, Bahrain",
    "me-west1": "Dubai, UAE",
    "northamerica-northeast1": "Montréal, QC, Canada",
    "northamerica-northeast2": "Toronto, ON, Canada",
    "southamerica-east1": "Sao Paulo, Brazil",
    "southamerica-west1": "Santiago, Chile",
    "us-central1": "Council Bluffs, IA, USA",
    "us-east1": "Moncks Corner, SC, USA",
    "us-east4": "Ashburn, VA, USA",
    "us-east5": "Atlanta, GA, USA",
    "us-south1": "Louisiana, USA",
    "us-west1": "The Dalles, OR, USA",
    "us-west2": "Los Angeles, CA, USA",
    "us-west3": "Salt Lake City, UT, USA",
    "us-west4": "Las Vegas, NV, USA",
}

geolocator = Nominatim(user_agent="geoapiExercises")


@memory.cache
def get_distance_to_city(city: str, region: str) -> float:
    """
    Many free API services implement rate limiting to prevent abuse and
    ensure fair usage. If you make requests too quickly, the service may
    temporarily or permanently block your IP address.
    """
    geolocator = Nominatim(user_agent="python-script")
    city_location = geolocator.geocode(city)
    time.sleep(0.5)
    region_location = geolocator.geocode(region_locations[region])
    time.sleep(0.5)
    return geodesic(
        (city_location.latitude, city_location.longitude),
        (region_location.latitude, region_location.longitude),
    ).kilometers


def sort_regions_by_distance(city):
    regions_distance = {
        region: get_distance_to_city(city, region) for region in tqdm(region_locations)
    }
    regions_sorted = sorted(regions_distance.items(), key=operator.itemgetter(1))
    return regions_sorted


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
    city = "Toronto, ON, Canada"
    machine_type = "n1-standard-16"
    regions = sort_regions_by_distance(city)
    instance_name = get_next_instance_name(
        project_id
    )  # Replace with your instance name

    for region in regions:
        zones = get_zones(project_id, region)
        for zone in zones:
            success = create_instance(
                project_id, zone, instance_name, machine_type, "cuda-support"
            )
            if success:
                return


if __name__ == "__main__":
    main()
