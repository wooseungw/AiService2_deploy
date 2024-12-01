import os
import subprocess
from azure.identity import DefaultAzureCredential
from azure.mgmt.resource import ResourceManagementClient
from azure.mgmt.containerregistry import ContainerRegistryManagementClient
from azure.mgmt.web import WebSiteManagementClient
from azure.mgmt.web.models import (
    AppServicePlan,
    SkuDescription,
    Site,
    SiteConfig,
)
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

def build_and_push_docker_image(acr_name, image_name, dockerfile_path):
    """
    Docker 이미지를 빌드하고 ACR에 푸시하는 함수입니다.
    """
    # Docker 이미지 빌드
    subprocess.run(["docker", "build", "-t", image_name, "--file", dockerfile_path, "."], check=True)

    # Azure CLI를 사용하여 ACR 로그인
    subprocess.run(["az", "acr", "login", "--name", acr_name], check=True)

    # Docker 이미지 푸시
    subprocess.run(["docker", "push", image_name], check=True)

def main():
    # 환경 변수 설정
    subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")
    resource_group_name = "myresourcegroup"
    location = "eastus"
    acr_name = "myacrname"
    app_service_plan_name = "myAppServicePlan"
    web_app_name = "mywebapp"
    image_name = f"{acr_name}.azurecr.io/myapp:latest"
    dockerfile_path = ".Dockerfile"

    # 인증 설정
    credential = DefaultAzureCredential()

    # 리소스 그룹 클라이언트 및 생성
    resource_client = ResourceManagementClient(credential, subscription_id)
    resource_client.resource_groups.create_or_update(resource_group_name, {"location": location})

    # ACR 클라이언트 및 생성
    acr_client = ContainerRegistryManagementClient(credential, subscription_id)
    acr_client.registries.begin_create(
        resource_group_name, acr_name, {"location": location, "sku": {"name": "Basic"}}
    ).result()

    # ACR 로그인 서버 가져오기
    acr_login_server = acr_client.registries.get(resource_group_name, acr_name).login_server

    # Docker 이미지 빌드 및 푸시
    build_and_push_docker_image(acr_name, image_name, dockerfile_path)

    # App Service 플랜 클라이언트 및 생성
    web_client = WebSiteManagementClient(credential, subscription_id)
    app_service_plan = web_client.app_service_plans.begin_create_or_update(
        resource_group_name,
        app_service_plan_name,
        AppServicePlan(
            location=location, sku=SkuDescription(name="B1", capacity=1), reserved=True
        ),
    ).result()

    # Web App 생성 및 Docker 컨테이너 이미지 설정
    web_app = web_client.web_apps.begin_create_or_update(
        resource_group_name,
        web_app_name,
        Site(
            location=location,
            server_farm_id=app_service_plan.id,
            site_config=SiteConfig(
                app_settings=[
                    {"name": "DOCKER_REGISTRY_SERVER_URL", "value": f"https://{acr_login_server}"},
                    {"name": "DOCKER_REGISTRY_SERVER_USERNAME", "value": os.getenv("AZURE_REGISTRY_USERNAME")},
                    {"name": "DOCKER_REGISTRY_SERVER_PASSWORD", "value": os.getenv("AZURE_REGISTRY_PASSWORD")},
                ],
                linux_fx_version=f"DOCKER|{image_name}",
            ),
        ),
    ).result()

    print(f"Web App deployed at https://{web_app_name}.azurewebsites.net")

if __name__ == "__main__":
    main()