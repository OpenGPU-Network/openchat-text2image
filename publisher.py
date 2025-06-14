from web3 import Web3

from ogpu.client import ChainConfig, ChainId, publish_source, SourceInfo, ImageEnvironments, DeliveryMethod



ChainConfig.set_chain(chain_id=ChainId.OGPU_MAINNET)

source_info = SourceInfo(
    name="OpenChat - text2image",
    description="The AI model that powers the text2image functionality of the OpenChat bot.",
    logoUrl="",
    imageEnvs= ImageEnvironments(
        nvidia="https://raw.githubusercontent.com/OpenGPU-Network/openchat-text2image/refs/heads/main/docker-compose/nvidia.yml",
    ),
    minPayment=Web3.to_wei(0.001, "ether"),
    minAvailableLockup=Web3.to_wei(0, "ether"),
    maxExpiryDuration=86400,  # 24 hour in seconds
    deliveryMethod=DeliveryMethod.FIRST_RESPONSE,
)

source_address = publish_source(source_info)
print(f"Source published successfully at: {source_address}")