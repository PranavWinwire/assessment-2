from azure.storage.blob import BlobServiceClient, ContentSettings
from azure.core.exceptions import ResourceExistsError
from app.core.config import settings
from app.core.exceptions import DocumentNotFoundError, StorageError


def get_blob_service_client():
    return BlobServiceClient.from_connection_string(
        settings.azure_storage_connection_string
    )


def get_container_client():
    return get_blob_service_client().get_container_client(
        settings.azure_storage_container
    )


def upload_blob(blob_name: str, data: bytes, content_type: str) -> None:
    """Upload or replace a blob and attach its media type."""
    try:
        container = get_container_client()
        try:
            container.create_container()
        except ResourceExistsError:
            pass
        container.upload_blob(
            name=blob_name,
            data=data,
            overwrite=True,
            content_settings=ContentSettings(content_type=content_type),
        )
    except Exception as exc:
        raise StorageError("The document could not be stored.") from exc


def download_blob(blob_name: str) -> bytes:
    """Download a blob's bytes, mapping missing blobs to a safe domain error."""
    try:
        return get_container_client().download_blob(blob_name).readall()
    except Exception as exc:
        if getattr(exc, "status_code", None) == 404:
            raise DocumentNotFoundError("The referenced document was not found.") from exc
        raise StorageError("Blob Storage is unavailable.") from exc
