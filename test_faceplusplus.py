import os
import time
import requests
from json import JSONDecoder
from tqdm import tqdm
from retrying import retry

@retry(wait_exponential_multiplier=1000, wait_exponential_max=10000, stop_max_attempt_number=5)
def make_api_request(http_url, data, files):
    response = requests.post(http_url, data=data, files=files)
    response.raise_for_status()  # Raise an HTTPError for bad responses
    return response

def attack_faceplusplus(save_path, clean_path, target_path, api_key, api_secret, attack=True):
    http_url = 'https://api-us.faceplusplus.com/facepp/v3/compare'
    data = {"api_key": api_key, "api_secret": api_secret}
    target_name = os.path.abspath(target_path)
    confidence = 0
    total = 0

    if attack:
        for ref in os.listdir(save_path):
            path = os.path.join(save_path, ref)
            for img in tqdm(os.listdir(path), desc='Faceplusplus ' + ref):
                source_name = os.path.join(path, img)
                files = {"image_file1": open(source_name, "rb"), "image_file2": open(target_name, "rb")}
                try:
                    time.sleep(1)
                    response = make_api_request(http_url, data, files)
                    req_con = response.content.decode('utf-8')
                    req_dict = JSONDecoder().decode(req_con)
                    if 'confidence' in req_dict.keys():
                        confidence += req_dict['confidence']
                        total += 1
                    else:
                        print(f"Error in response: {req_dict}")
                except requests.exceptions.RequestException as e:
                    print(f"Request failed: {e}")
        print('Is Adversarial Attack:', attack)
        print('Face++ confidence: {:.4f}'.format(confidence / total if total > 0 else 0))
    else:
        for img in tqdm(os.listdir(clean_path), desc='Faceplusplus clean'):
            source_name = os.path.join(clean_path, img)
            files = {"image_file1": open(source_name, "rb"), "image_file2": open(target_name, "rb")}
            try:
                time.sleep(0.5)
                response = make_api_request(http_url, data, files)
                req_con = response.content.decode('utf-8')
                req_dict = JSONDecoder().decode(req_con)
                if 'confidence' in req_dict.keys():
                    confidence += req_dict['confidence']
                    total += 1
                else:
                    print(f"Error in response: {req_dict}")
            except requests.exceptions.RequestException as e:
                print(f"Request failed: {e}")
        print('Is Adversarial Attack:', attack)
        print('Face++ confidence: {:.4f}'.format(confidence / total if total > 0 else 0))


if __name__ == '__main__':
    save_path = "assets/datasets/save"  # replace with your path to generated images
    clean_path = "assets/datasets/CelebA-HQ"  # replace with your path to clean images
    target_path = "assets/datasets/test/047073.jpg"  # replace with your path to target image
    api_key = "xxxx"  # replace with your Face++ API key
    api_secret = "xxxx"  # replace with your Face++ API secret

    attack_faceplusplus(save_path, clean_path, target_path, api_key, api_secret, attack=False)
    attack_faceplusplus(save_path, clean_path, target_path, api_key, api_secret, attack=True)
