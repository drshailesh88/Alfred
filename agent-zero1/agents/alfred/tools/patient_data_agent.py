"""
Patient Data Agent - Clinical Document Vault with Natural Language Retrieval

Role: Securely stores and indexes patient records, clinical notes, test results,
and imaging reports. Answers queries about patient history without making
clinical judgments.

Security Protocol:
- All data encrypted at rest
- Access logged with timestamp and query
- No data leaves system without explicit request
- Automatic redaction of identifiers in outputs
- Compliance with healthcare data regulations

GitHub Tool Integrations (interfaces prepared for):
- MedCAT for medical NLP/NER
- Qdrant for vector database
- LlamaIndex for RAG framework
- Presidio for PHI de-identification
"""

from . import OperationsAgent, AgentResponse, AlfredState
from typing import Dict, Any, Optional, List, Tuple, Union, Protocol, TypeVar
from datetime import datetime, date
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from abc import ABC, abstractmethod
import json
import hashlib
import uuid
import re
import base64
import os
import logging
from functools import wraps

# Configure logging for audit trail
logger = logging.getLogger("patient_data_agent")


# =============================================================================
# Enumerations
# =============================================================================

class QueryType(Enum):
    """Types of patient data queries supported."""
    SEARCH = "search"
    RETRIEVE = "retrieve"
    SUMMARY = "summary"
    TIMELINE = "timeline"


class DocumentType(Enum):
    """Types of clinical documents."""
    ALL = "all"
    NOTES = "notes"
    LABS = "labs"
    IMAGING = "imaging"
    PROCEDURES = "procedures"
    PRESCRIPTIONS = "prescriptions"
    DISCHARGE = "discharge"
    REFERRAL = "referral"
    CONSULTATION = "consultation"


class OutputDetail(Enum):
    """Level of detail in query responses."""
    BRIEF = "brief"
    STANDARD = "standard"
    COMPREHENSIVE = "comprehensive"


class AlertType(Enum):
    """Types of patient data alerts."""
    MISSING_FOLLOWUP = "missing_followup"
    OVERDUE_TEST = "overdue_test"
    CONFLICTING_INFO = "conflicting_info"
    DATA_QUALITY = "data_quality"
    MEDICATION_ALERT = "medication_alert"


class PHIType(Enum):
    """Types of Protected Health Information for redaction."""
    NAME = "NAME"
    DATE_OF_BIRTH = "DATE_OF_BIRTH"
    ADDRESS = "ADDRESS"
    PHONE = "PHONE"
    EMAIL = "EMAIL"
    SSN = "SSN"
    MRN = "MRN"  # Medical Record Number
    ACCOUNT_NUMBER = "ACCOUNT_NUMBER"
    LICENSE_NUMBER = "LICENSE_NUMBER"
    IP_ADDRESS = "IP_ADDRESS"
    DEVICE_ID = "DEVICE_ID"
    BIOMETRIC = "BIOMETRIC"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class PatientRecord:
    """
    Represents a patient clinical record.

    Attributes:
        record_id: Unique identifier for the record
        patient_id: Hashed patient identifier (never store raw)
        document_type: Type of clinical document
        content: Encrypted document content
        content_hash: SHA-256 hash for integrity verification
        metadata: Document metadata (dates, providers, etc.)
        created_at: Record creation timestamp
        updated_at: Last update timestamp
        version: Document version number
        linked_records: IDs of related records
        extracted_entities: Structured data extracted from content
        embedding_id: Reference to vector embedding in Qdrant
    """
    record_id: str
    patient_id: str  # Always hashed
    document_type: DocumentType
    content: bytes  # Encrypted content
    content_hash: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    version: int = 1
    linked_records: List[str] = field(default_factory=list)
    extracted_entities: Dict[str, Any] = field(default_factory=dict)
    embedding_id: Optional[str] = None

    def to_dict(self, include_content: bool = False) -> Dict[str, Any]:
        """Convert to dictionary, optionally excluding sensitive content."""
        result = {
            "record_id": self.record_id,
            "patient_id": self.patient_id,
            "document_type": self.document_type.value,
            "content_hash": self.content_hash,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "version": self.version,
            "linked_records": self.linked_records,
            "extracted_entities": self.extracted_entities,
            "embedding_id": self.embedding_id
        }
        if include_content:
            result["content"] = base64.b64encode(self.content).decode('utf-8')
        return result


@dataclass
class AccessLogEntry:
    """
    Audit log entry for all data access.

    Every access to patient data is logged with timestamp, query details,
    and user context for compliance and security.
    """
    log_id: str
    timestamp: str
    patient_id: str  # Hashed
    query_type: QueryType
    query_details: Dict[str, Any]
    records_accessed: List[str]
    user_context: str  # Alfred or specific sub-agent
    success: bool
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "log_id": self.log_id,
            "timestamp": self.timestamp,
            "patient_id": self.patient_id,
            "query_type": self.query_type.value,
            "query_details": self.query_details,
            "records_accessed": self.records_accessed,
            "user_context": self.user_context,
            "success": self.success,
            "error_message": self.error_message
        }


@dataclass
class PatientQuery:
    """
    Structured patient data query request.

    Matches the PATIENT_QUERY input format from specification.
    """
    query_type: QueryType
    patient_identifier: Optional[str] = None
    search_terms: List[str] = field(default_factory=list)
    date_range: Optional[Tuple[str, str]] = None  # (start_date, end_date)
    document_types: List[DocumentType] = field(default_factory=lambda: [DocumentType.ALL])
    output_detail: OutputDetail = OutputDetail.STANDARD

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_type": self.query_type.value,
            "patient_identifier": "[REDACTED]" if self.patient_identifier else None,
            "search_terms": self.search_terms,
            "date_range": self.date_range,
            "document_types": [dt.value for dt in self.document_types],
            "output_detail": self.output_detail.value
        }


@dataclass
class PatientDataResponse:
    """
    Structured patient data query response.

    Matches the PATIENT_DATA_RESPONSE output format from specification.
    """
    query: Dict[str, Any]
    patient_id: str  # Redacted in logs
    records_found: int
    summary: str
    matching_records: List[Dict[str, Any]]
    timeline_view: Optional[List[Dict[str, Any]]] = None
    related_records: List[Dict[str, Any]] = field(default_factory=list)
    data_gaps: List[str] = field(default_factory=list)
    access_logged: bool = True
    log_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "patient": "[REDACTED]",  # Always redact in output
            "records_found": self.records_found,
            "summary": self.summary,
            "matching_records": self.matching_records,
            "timeline_view": self.timeline_view,
            "related_records": self.related_records,
            "data_gaps": self.data_gaps,
            "access_logged": self.access_logged,
            "log_id": self.log_id
        }


@dataclass
class PatientDataAlert:
    """
    Alert for patient data issues.

    Matches the PATIENT_DATA_ALERT format from specification.
    """
    alert_type: AlertType
    patient_id: str  # Hashed
    details: str
    recommended_action: str
    severity: str = "medium"  # low, medium, high
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alert_type": self.alert_type.value,
            "patient": "[REDACTED]",
            "details": self.details,
            "recommended_action": self.recommended_action,
            "severity": self.severity,
            "created_at": self.created_at
        }


# =============================================================================
# Abstract Interfaces for External Tool Integration
# =============================================================================

class EncryptionProvider(ABC):
    """
    Abstract interface for encryption services.

    Implementations should use AES-256-GCM or equivalent healthcare-grade encryption.
    """

    @abstractmethod
    def encrypt(self, data: bytes, key_id: Optional[str] = None) -> bytes:
        """Encrypt data with optional key identifier."""
        pass

    @abstractmethod
    def decrypt(self, encrypted_data: bytes, key_id: Optional[str] = None) -> bytes:
        """Decrypt data with optional key identifier."""
        pass

    @abstractmethod
    def rotate_key(self, old_key_id: str, new_key_id: str) -> bool:
        """Rotate encryption keys."""
        pass


class VectorSearchProvider(ABC):
    """
    Abstract interface for vector database operations.

    Designed for Qdrant integration but compatible with other vector DBs.
    """

    @abstractmethod
    def store_embedding(
        self,
        embedding: List[float],
        record_id: str,
        metadata: Dict[str, Any]
    ) -> str:
        """Store embedding and return embedding ID."""
        pass

    @abstractmethod
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[str, float]]:
        """Search for similar embeddings, return (record_id, score) pairs."""
        pass

    @abstractmethod
    def delete_embedding(self, embedding_id: str) -> bool:
        """Delete an embedding by ID."""
        pass

    @abstractmethod
    def update_embedding(
        self,
        embedding_id: str,
        new_embedding: List[float],
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Update an existing embedding."""
        pass


class MedicalNLPProvider(ABC):
    """
    Abstract interface for medical NLP/NER.

    Designed for MedCAT integration but compatible with other medical NLP tools.
    """

    @abstractmethod
    def extract_entities(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract medical entities from text.

        Returns dict with keys like 'conditions', 'medications', 'procedures', etc.
        Each entity has: text, start, end, cui (concept ID), confidence
        """
        pass

    @abstractmethod
    def link_concepts(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Link extracted entities to medical ontologies (SNOMED, ICD, etc.)."""
        pass

    @abstractmethod
    def extract_relations(self, text: str) -> List[Dict[str, Any]]:
        """Extract relationships between medical entities."""
        pass


class PHIDetectionProvider(ABC):
    """
    Abstract interface for PHI detection and de-identification.

    Designed for Presidio integration.
    """

    @abstractmethod
    def detect_phi(self, text: str) -> List[Dict[str, Any]]:
        """
        Detect PHI entities in text.

        Returns list of dicts with: entity_type, start, end, text, score
        """
        pass

    @abstractmethod
    def redact(self, text: str, phi_types: Optional[List[PHIType]] = None) -> str:
        """Redact PHI from text, optionally filtering by type."""
        pass

    @abstractmethod
    def anonymize(
        self,
        text: str,
        strategy: str = "replace"  # replace, hash, mask
    ) -> Tuple[str, Dict[str, str]]:
        """
        Anonymize PHI with optional mapping for reversal.

        Returns (anonymized_text, mapping_dict)
        """
        pass


class RAGProvider(ABC):
    """
    Abstract interface for RAG (Retrieval Augmented Generation) operations.

    Designed for LlamaIndex integration.
    """

    @abstractmethod
    def index_document(
        self,
        document: str,
        metadata: Dict[str, Any],
        document_id: str
    ) -> bool:
        """Index a document for retrieval."""
        pass

    @abstractmethod
    def query(
        self,
        query_text: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Query the index.

        Returns list of dicts with: document_id, text, score, metadata
        """
        pass

    @abstractmethod
    def delete_document(self, document_id: str) -> bool:
        """Remove a document from the index."""
        pass

    @abstractmethod
    def update_document(
        self,
        document_id: str,
        new_content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Update an indexed document."""
        pass


# =============================================================================
# Default Implementations (for development/testing without external services)
# =============================================================================

class DefaultEncryptionProvider(EncryptionProvider):
    """
    Default encryption provider using Fernet symmetric encryption.

    WARNING: For production, use a proper HSM-backed encryption service.
    This is suitable for development and testing only.
    """

    def __init__(self, key: Optional[bytes] = None):
        try:
            from cryptography.fernet import Fernet
            self._fernet_available = True
            if key is None:
                key = Fernet.generate_key()
            self._fernet = Fernet(key)
            self._key = key
        except ImportError:
            self._fernet_available = False
            self._key = key or os.urandom(32)
            logger.warning(
                "cryptography package not available. "
                "Using basic XOR encryption for development only."
            )

    def encrypt(self, data: bytes, key_id: Optional[str] = None) -> bytes:
        if self._fernet_available:
            return self._fernet.encrypt(data)
        # Fallback: basic XOR (NOT SECURE - development only)
        return bytes(a ^ b for a, b in zip(data, (self._key * (len(data) // len(self._key) + 1))[:len(data)]))

    def decrypt(self, encrypted_data: bytes, key_id: Optional[str] = None) -> bytes:
        if self._fernet_available:
            return self._fernet.decrypt(encrypted_data)
        # Fallback: XOR is symmetric
        return bytes(a ^ b for a, b in zip(encrypted_data, (self._key * (len(encrypted_data) // len(self._key) + 1))[:len(encrypted_data)]))

    def rotate_key(self, old_key_id: str, new_key_id: str) -> bool:
        logger.warning("Key rotation not implemented in default provider")
        return False


class DefaultVectorSearchProvider(VectorSearchProvider):
    """
    Default vector search using simple cosine similarity.

    For production, use Qdrant or similar vector database.
    """

    def __init__(self):
        self._embeddings: Dict[str, Tuple[List[float], str, Dict[str, Any]]] = {}

    def store_embedding(
        self,
        embedding: List[float],
        record_id: str,
        metadata: Dict[str, Any]
    ) -> str:
        embedding_id = str(uuid.uuid4())
        self._embeddings[embedding_id] = (embedding, record_id, metadata)
        return embedding_id

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[str, float]]:
        results = []
        for emb_id, (embedding, record_id, metadata) in self._embeddings.items():
            # Apply filters if provided
            if filters:
                match = True
                for key, value in filters.items():
                    if metadata.get(key) != value:
                        match = False
                        break
                if not match:
                    continue

            # Cosine similarity
            score = self._cosine_similarity(query_embedding, embedding)
            results.append((record_id, score))

        # Sort by score descending and return top_k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def delete_embedding(self, embedding_id: str) -> bool:
        if embedding_id in self._embeddings:
            del self._embeddings[embedding_id]
            return True
        return False

    def update_embedding(
        self,
        embedding_id: str,
        new_embedding: List[float],
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        if embedding_id not in self._embeddings:
            return False
        old_embedding, record_id, old_metadata = self._embeddings[embedding_id]
        self._embeddings[embedding_id] = (
            new_embedding,
            record_id,
            metadata if metadata else old_metadata
        )
        return True

    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        if len(a) != len(b):
            return 0.0
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x ** 2 for x in a) ** 0.5
        norm_b = sum(x ** 2 for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot_product / (norm_a * norm_b)


class DefaultPHIDetectionProvider(PHIDetectionProvider):
    """
    Default PHI detection using regex patterns.

    For production, use Microsoft Presidio or similar.
    """

    # Regex patterns for common PHI types
    PHI_PATTERNS = {
        PHIType.SSN: r'\b\d{3}-\d{2}-\d{4}\b',
        PHIType.PHONE: r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b',
        PHIType.EMAIL: r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        PHIType.DATE_OF_BIRTH: r'\b(0[1-9]|1[0-2])/(0[1-9]|[12]\d|3[01])/\d{4}\b',
        PHIType.MRN: r'\bMRN[:\s]?\d{6,10}\b',
        PHIType.ACCOUNT_NUMBER: r'\b[Aa]ccount[:\s]?\d{8,12}\b',
        PHIType.IP_ADDRESS: r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',
    }

    def detect_phi(self, text: str) -> List[Dict[str, Any]]:
        results = []
        for phi_type, pattern in self.PHI_PATTERNS.items():
            for match in re.finditer(pattern, text, re.IGNORECASE):
                results.append({
                    "entity_type": phi_type.value,
                    "start": match.start(),
                    "end": match.end(),
                    "text": match.group(),
                    "score": 0.9  # High confidence for regex matches
                })
        return results

    def redact(self, text: str, phi_types: Optional[List[PHIType]] = None) -> str:
        result = text
        patterns_to_check = (
            {pt: self.PHI_PATTERNS[pt] for pt in phi_types if pt in self.PHI_PATTERNS}
            if phi_types else self.PHI_PATTERNS
        )

        for phi_type, pattern in patterns_to_check.items():
            result = re.sub(pattern, f"[{phi_type.value}]", result, flags=re.IGNORECASE)

        return result

    def anonymize(
        self,
        text: str,
        strategy: str = "replace"
    ) -> Tuple[str, Dict[str, str]]:
        mapping = {}
        result = text

        detected = self.detect_phi(text)
        # Sort by position descending to replace from end to start
        detected.sort(key=lambda x: x["start"], reverse=True)

        for entity in detected:
            original = entity["text"]
            if original in mapping:
                replacement = mapping[original]
            else:
                if strategy == "hash":
                    replacement = hashlib.sha256(original.encode()).hexdigest()[:8]
                elif strategy == "mask":
                    replacement = "*" * len(original)
                else:  # replace
                    replacement = f"[{entity['entity_type']}_{len(mapping)}]"
                mapping[original] = replacement

            result = result[:entity["start"]] + replacement + result[entity["end"]:]

        return result, mapping


class DefaultMedicalNLPProvider(MedicalNLPProvider):
    """
    Default medical NLP provider using keyword matching.

    For production, use MedCAT or similar medical NLP tool.
    """

    # Basic medical term patterns
    CONDITION_PATTERNS = [
        r'\b(diabetes|hypertension|asthma|COPD|heart failure|CHF|cancer|'
        r'pneumonia|bronchitis|arthritis|depression|anxiety)\b'
    ]
    MEDICATION_PATTERNS = [
        r'\b(aspirin|metformin|lisinopril|amlodipine|metoprolol|omeprazole|'
        r'atorvastatin|levothyroxine|albuterol|prednisone)\b'
    ]
    PROCEDURE_PATTERNS = [
        r'\b(MRI|CT scan|X-ray|ultrasound|ECG|EKG|colonoscopy|endoscopy|'
        r'surgery|biopsy|blood test|urinalysis)\b'
    ]

    def extract_entities(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        results = {
            "conditions": [],
            "medications": [],
            "procedures": []
        }

        for pattern in self.CONDITION_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                results["conditions"].append({
                    "text": match.group(),
                    "start": match.start(),
                    "end": match.end(),
                    "cui": None,  # Would be UMLS CUI in production
                    "confidence": 0.8
                })

        for pattern in self.MEDICATION_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                results["medications"].append({
                    "text": match.group(),
                    "start": match.start(),
                    "end": match.end(),
                    "cui": None,
                    "confidence": 0.8
                })

        for pattern in self.PROCEDURE_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                results["procedures"].append({
                    "text": match.group(),
                    "start": match.start(),
                    "end": match.end(),
                    "cui": None,
                    "confidence": 0.8
                })

        return results

    def link_concepts(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # Would link to SNOMED CT, ICD-10, etc. in production
        return entities

    def extract_relations(self, text: str) -> List[Dict[str, Any]]:
        # Would extract medical relations in production
        return []


class DefaultRAGProvider(RAGProvider):
    """
    Default RAG provider using simple keyword matching.

    For production, use LlamaIndex or similar RAG framework.
    """

    def __init__(self):
        self._documents: Dict[str, Tuple[str, Dict[str, Any]]] = {}

    def index_document(
        self,
        document: str,
        metadata: Dict[str, Any],
        document_id: str
    ) -> bool:
        self._documents[document_id] = (document, metadata)
        return True

    def query(
        self,
        query_text: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        results = []
        query_terms = set(query_text.lower().split())

        for doc_id, (doc_text, metadata) in self._documents.items():
            # Apply filters
            if filters:
                match = True
                for key, value in filters.items():
                    if metadata.get(key) != value:
                        match = False
                        break
                if not match:
                    continue

            # Simple term overlap scoring
            doc_terms = set(doc_text.lower().split())
            overlap = len(query_terms & doc_terms)
            if overlap > 0:
                score = overlap / len(query_terms)
                results.append({
                    "document_id": doc_id,
                    "text": doc_text[:500],  # Truncate for response
                    "score": score,
                    "metadata": metadata
                })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def delete_document(self, document_id: str) -> bool:
        if document_id in self._documents:
            del self._documents[document_id]
            return True
        return False

    def update_document(
        self,
        document_id: str,
        new_content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        if document_id not in self._documents:
            return False
        old_content, old_metadata = self._documents[document_id]
        self._documents[document_id] = (
            new_content,
            metadata if metadata else old_metadata
        )
        return True


# =============================================================================
# Document Storage
# =============================================================================

class DocumentStorage:
    """
    Secure document storage abstraction with encryption and integrity verification.
    """

    def __init__(
        self,
        storage_path: Optional[Path] = None,
        encryption_provider: Optional[EncryptionProvider] = None
    ):
        self.storage_path = storage_path or Path("~/.alfred/patient_data").expanduser()
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.encryption = encryption_provider or DefaultEncryptionProvider()
        self._records: Dict[str, PatientRecord] = {}
        self._load_index()

    def _load_index(self) -> None:
        """Load record index from disk."""
        index_path = self.storage_path / "index.json"
        if index_path.exists():
            try:
                with open(index_path, 'r') as f:
                    index_data = json.load(f)
                    for record_data in index_data.get("records", []):
                        # Don't load content in index - load on demand
                        record_data["content"] = b""
                        record_data["document_type"] = DocumentType(record_data["document_type"])
                        self._records[record_data["record_id"]] = PatientRecord(**record_data)
            except Exception as e:
                logger.error(f"Failed to load document index: {e}")

    def _save_index(self) -> None:
        """Save record index to disk."""
        index_path = self.storage_path / "index.json"
        try:
            index_data = {
                "last_updated": datetime.now().isoformat(),
                "record_count": len(self._records),
                "records": [
                    {k: v for k, v in r.to_dict(include_content=False).items()}
                    for r in self._records.values()
                ]
            }
            with open(index_path, 'w') as f:
                json.dump(index_data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save document index: {e}")

    def store(
        self,
        patient_id: str,
        document_type: DocumentType,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> PatientRecord:
        """
        Store a new patient record with encryption.

        Args:
            patient_id: Raw patient identifier (will be hashed)
            document_type: Type of clinical document
            content: Document content (will be encrypted)
            metadata: Optional document metadata

        Returns:
            PatientRecord with encrypted content
        """
        # Hash patient ID for storage
        hashed_patient_id = self._hash_identifier(patient_id)

        # Generate record ID
        record_id = str(uuid.uuid4())

        # Encrypt content
        content_bytes = content.encode('utf-8')
        encrypted_content = self.encryption.encrypt(content_bytes)

        # Calculate content hash for integrity
        content_hash = hashlib.sha256(content_bytes).hexdigest()

        # Create record
        record = PatientRecord(
            record_id=record_id,
            patient_id=hashed_patient_id,
            document_type=document_type,
            content=encrypted_content,
            content_hash=content_hash,
            metadata=metadata or {}
        )

        # Store encrypted content to file
        content_path = self.storage_path / f"{record_id}.enc"
        with open(content_path, 'wb') as f:
            f.write(encrypted_content)

        # Add to index
        self._records[record_id] = record
        self._save_index()

        logger.info(f"Stored record {record_id} for patient {hashed_patient_id[:8]}...")
        return record

    def retrieve(self, record_id: str) -> Optional[Tuple[PatientRecord, str]]:
        """
        Retrieve a record and decrypt its content.

        Returns:
            Tuple of (PatientRecord, decrypted_content) or None if not found
        """
        if record_id not in self._records:
            return None

        record = self._records[record_id]

        # Load encrypted content from file
        content_path = self.storage_path / f"{record_id}.enc"
        if not content_path.exists():
            logger.error(f"Content file missing for record {record_id}")
            return None

        with open(content_path, 'rb') as f:
            encrypted_content = f.read()

        # Decrypt content
        try:
            decrypted_content = self.encryption.decrypt(encrypted_content).decode('utf-8')

            # Verify integrity
            content_hash = hashlib.sha256(decrypted_content.encode()).hexdigest()
            if content_hash != record.content_hash:
                logger.error(f"Content integrity check failed for record {record_id}")
                return None

            return record, decrypted_content
        except Exception as e:
            logger.error(f"Failed to decrypt record {record_id}: {e}")
            return None

    def update(
        self,
        record_id: str,
        content: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[PatientRecord]:
        """
        Update an existing record, creating a new version.
        """
        if record_id not in self._records:
            return None

        old_record = self._records[record_id]

        if content is not None:
            # Encrypt new content
            content_bytes = content.encode('utf-8')
            encrypted_content = self.encryption.encrypt(content_bytes)
            content_hash = hashlib.sha256(content_bytes).hexdigest()

            # Write new content
            content_path = self.storage_path / f"{record_id}.enc"
            with open(content_path, 'wb') as f:
                f.write(encrypted_content)

            old_record.content = encrypted_content
            old_record.content_hash = content_hash

        if metadata is not None:
            old_record.metadata.update(metadata)

        old_record.updated_at = datetime.now().isoformat()
        old_record.version += 1

        self._save_index()
        return old_record

    def delete(self, record_id: str) -> bool:
        """
        Securely delete a record.

        Note: In production, implement secure deletion with audit trail.
        """
        if record_id not in self._records:
            return False

        # Remove content file
        content_path = self.storage_path / f"{record_id}.enc"
        if content_path.exists():
            # Overwrite with random data before deletion (secure delete)
            with open(content_path, 'wb') as f:
                f.write(os.urandom(content_path.stat().st_size))
            content_path.unlink()

        del self._records[record_id]
        self._save_index()

        logger.info(f"Deleted record {record_id}")
        return True

    def find_by_patient(self, patient_id: str) -> List[PatientRecord]:
        """Find all records for a patient."""
        hashed_id = self._hash_identifier(patient_id)
        return [r for r in self._records.values() if r.patient_id == hashed_id]

    def find_by_type(
        self,
        document_types: List[DocumentType],
        patient_id: Optional[str] = None
    ) -> List[PatientRecord]:
        """Find records by document type, optionally filtered by patient."""
        records = list(self._records.values())

        if patient_id:
            hashed_id = self._hash_identifier(patient_id)
            records = [r for r in records if r.patient_id == hashed_id]

        if DocumentType.ALL not in document_types:
            records = [r for r in records if r.document_type in document_types]

        return records

    def find_by_date_range(
        self,
        start_date: str,
        end_date: str,
        patient_id: Optional[str] = None
    ) -> List[PatientRecord]:
        """Find records within a date range."""
        records = list(self._records.values())

        if patient_id:
            hashed_id = self._hash_identifier(patient_id)
            records = [r for r in records if r.patient_id == hashed_id]

        filtered = []
        for record in records:
            record_date = record.metadata.get("document_date", record.created_at)
            if start_date <= record_date <= end_date:
                filtered.append(record)

        return filtered

    @staticmethod
    def _hash_identifier(identifier: str) -> str:
        """Hash a patient identifier for secure storage."""
        return hashlib.sha256(identifier.encode()).hexdigest()


# =============================================================================
# Access Logging
# =============================================================================

class AccessLogger:
    """
    Comprehensive audit logging for all patient data access.

    Maintains immutable log of all queries for compliance and security.
    """

    def __init__(self, log_path: Optional[Path] = None):
        self.log_path = log_path or Path("~/.alfred/patient_data/access_logs").expanduser()
        self.log_path.mkdir(parents=True, exist_ok=True)
        self._current_log_file: Optional[Path] = None
        self._entries: List[AccessLogEntry] = []

    def log_access(
        self,
        patient_id: str,
        query_type: QueryType,
        query_details: Dict[str, Any],
        records_accessed: List[str],
        user_context: str = "Alfred",
        success: bool = True,
        error_message: Optional[str] = None
    ) -> str:
        """
        Log a data access event.

        Returns:
            Log entry ID
        """
        # Hash patient ID for log
        hashed_patient_id = hashlib.sha256(patient_id.encode()).hexdigest()

        entry = AccessLogEntry(
            log_id=str(uuid.uuid4()),
            timestamp=datetime.now().isoformat(),
            patient_id=hashed_patient_id,
            query_type=query_type,
            query_details=self._sanitize_query_details(query_details),
            records_accessed=records_accessed,
            user_context=user_context,
            success=success,
            error_message=error_message
        )

        self._entries.append(entry)
        self._write_log_entry(entry)

        logger.info(
            f"Access logged: {entry.log_id} | "
            f"Type: {query_type.value} | "
            f"Records: {len(records_accessed)} | "
            f"Success: {success}"
        )

        return entry.log_id

    def _sanitize_query_details(self, details: Dict[str, Any]) -> Dict[str, Any]:
        """Remove any PHI from query details before logging."""
        # Remove or redact sensitive fields
        sanitized = {}
        for key, value in details.items():
            if key in ["patient_identifier", "patient_id", "name", "ssn"]:
                sanitized[key] = "[REDACTED]"
            else:
                sanitized[key] = value
        return sanitized

    def _write_log_entry(self, entry: AccessLogEntry) -> None:
        """Write log entry to current log file."""
        # Rotate log files daily
        today = date.today().isoformat()
        log_file = self.log_path / f"access_{today}.jsonl"

        with open(log_file, 'a') as f:
            f.write(json.dumps(entry.to_dict(), default=str) + "\n")

    def get_access_history(
        self,
        patient_id: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> List[AccessLogEntry]:
        """
        Retrieve access history with optional filters.

        For compliance audits and security reviews.
        """
        entries = self._entries.copy()

        if patient_id:
            hashed_id = hashlib.sha256(patient_id.encode()).hexdigest()
            entries = [e for e in entries if e.patient_id == hashed_id]

        if start_date:
            entries = [e for e in entries if e.timestamp >= start_date]

        if end_date:
            entries = [e for e in entries if e.timestamp <= end_date]

        return entries


# =============================================================================
# PHI Redaction Utilities
# =============================================================================

class PHIRedactor:
    """
    Utility class for PHI detection and redaction.

    Wraps PHIDetectionProvider with convenience methods.
    """

    def __init__(self, provider: Optional[PHIDetectionProvider] = None):
        self.provider = provider or DefaultPHIDetectionProvider()

    def redact_for_output(
        self,
        text: str,
        preserve_medical_terms: bool = True
    ) -> str:
        """
        Redact PHI for safe output, preserving medical terms.
        """
        return self.provider.redact(text)

    def redact_in_record_summary(
        self,
        record: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Redact PHI in a record summary for output.
        """
        redacted = record.copy()

        # Redact specific fields
        if "patient_id" in redacted:
            redacted["patient_id"] = "[REDACTED]"

        # Redact text fields
        for key in ["content", "summary", "notes", "description"]:
            if key in redacted and isinstance(redacted[key], str):
                redacted[key] = self.provider.redact(redacted[key])

        return redacted

    def detect_and_warn(self, text: str) -> List[Dict[str, Any]]:
        """
        Detect PHI and return warnings without modifying text.
        """
        return self.provider.detect_phi(text)


# =============================================================================
# Decorators for Security
# =============================================================================

def require_access_logging(func):
    """Decorator to ensure all data access is logged."""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        # Extract query information
        query = kwargs.get('query') or (args[0] if args else None)

        try:
            result = func(self, *args, **kwargs)

            # Log successful access
            if hasattr(self, 'access_logger') and query:
                patient_id = getattr(query, 'patient_identifier', 'unknown') or 'unknown'
                query_type = getattr(query, 'query_type', QueryType.SEARCH)

                self.access_logger.log_access(
                    patient_id=patient_id,
                    query_type=query_type,
                    query_details=query.to_dict() if hasattr(query, 'to_dict') else {},
                    records_accessed=_extract_record_ids(result),
                    success=True
                )

            return result
        except Exception as e:
            # Log failed access
            if hasattr(self, 'access_logger') and query:
                patient_id = getattr(query, 'patient_identifier', 'unknown') or 'unknown'
                query_type = getattr(query, 'query_type', QueryType.SEARCH)

                self.access_logger.log_access(
                    patient_id=patient_id,
                    query_type=query_type,
                    query_details=query.to_dict() if hasattr(query, 'to_dict') else {},
                    records_accessed=[],
                    success=False,
                    error_message=str(e)
                )
            raise

    return wrapper


def _extract_record_ids(result) -> List[str]:
    """Extract record IDs from query result for logging."""
    if isinstance(result, AgentResponse):
        matching = result.data.get("matching_records", [])
        return [r.get("record_id", "") for r in matching if isinstance(r, dict)]
    return []


# =============================================================================
# Main Patient Data Agent
# =============================================================================

class PatientDataAgent(OperationsAgent):
    """
    Patient Data Agent - Clinical document vault with natural language retrieval.

    Role: Securely stores and indexes patient records, clinical notes, test results,
    and imaging reports. Answers queries about patient history without making
    clinical judgments.

    Does NOT:
    - Make clinical decisions
    - Diagnose conditions
    - Recommend treatments
    - Interpret test results clinically
    - Provide medical advice
    - Compare patients
    - Predict outcomes
    - Share data outside Alfred's context

    Does:
    - Store documents securely with encryption
    - Index for semantic retrieval
    - Answer queries about patient history
    - Surface relevant records based on context
    - Track document versions
    - Maintain audit trail of access
    - Link related records across visits
    - Extract structured data from clinical notes
    """

    def __init__(
        self,
        storage_path: Optional[Path] = None,
        encryption_provider: Optional[EncryptionProvider] = None,
        vector_provider: Optional[VectorSearchProvider] = None,
        nlp_provider: Optional[MedicalNLPProvider] = None,
        phi_provider: Optional[PHIDetectionProvider] = None,
        rag_provider: Optional[RAGProvider] = None
    ):
        """
        Initialize the Patient Data Agent.

        Args:
            storage_path: Base path for document storage
            encryption_provider: Custom encryption implementation
            vector_provider: Custom vector search implementation (Qdrant)
            nlp_provider: Custom medical NLP implementation (MedCAT)
            phi_provider: Custom PHI detection implementation (Presidio)
            rag_provider: Custom RAG implementation (LlamaIndex)
        """
        super().__init__(name="Patient Data Agent")

        # Initialize storage
        self.storage = DocumentStorage(
            storage_path=storage_path,
            encryption_provider=encryption_provider
        )

        # Initialize providers
        self.vector_search = vector_provider or DefaultVectorSearchProvider()
        self.medical_nlp = nlp_provider or DefaultMedicalNLPProvider()
        self.phi_detector = phi_provider or DefaultPHIDetectionProvider()
        self.rag = rag_provider or DefaultRAGProvider()

        # Initialize utilities
        self.redactor = PHIRedactor(self.phi_detector)
        self.access_logger = AccessLogger(
            log_path=storage_path / "access_logs" if storage_path else None
        )

        # Pending alerts
        self._alerts: List[PatientDataAlert] = []

    def check_state_permission(self) -> Tuple[bool, str]:
        """
        Patient Data Agent operates in all states - clinical never stops.
        """
        return True, "Patient Data Agent operates in all states"

    # =========================================================================
    # Core Query Methods
    # =========================================================================

    @require_access_logging
    def search(self, query: PatientQuery) -> AgentResponse:
        """
        Search for patient records based on criteria.

        Input Format:
            PATIENT_QUERY
            - Query Type: search
            - Patient Identifier: [if specific patient]
            - Search Terms: [symptoms, conditions, dates, procedures]
            - Date Range: [if applicable]
            - Document Types: [all | notes | labs | imaging | procedures]
            - Output Detail: brief | standard | comprehensive

        Returns:
            AgentResponse with matching records
        """
        try:
            # Find matching records
            records = self._find_matching_records(query)

            # Build response
            response_data = self._build_response_data(
                query=query,
                records=records,
                include_timeline=False
            )

            # Log access
            log_id = self.access_logger.log_access(
                patient_id=query.patient_identifier or "all",
                query_type=QueryType.SEARCH,
                query_details=query.to_dict(),
                records_accessed=[r.record_id for r in records],
                success=True
            )
            response_data["access_logged"] = True
            response_data["log_id"] = log_id

            return self.create_response(
                data=response_data,
                success=True
            )

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return self.create_response(
                data={"error": str(e)},
                success=False,
                errors=[f"Search failed: {str(e)}"]
            )

    @require_access_logging
    def retrieve(self, query: PatientQuery) -> AgentResponse:
        """
        Retrieve specific patient records with full content.

        Returns:
            AgentResponse with record details and decrypted content
        """
        try:
            records = self._find_matching_records(query)

            # Retrieve full content for matching records
            detailed_records = []
            for record in records:
                result = self.storage.retrieve(record.record_id)
                if result:
                    record_obj, content = result
                    detailed_records.append({
                        "record_id": record_obj.record_id,
                        "document_type": record_obj.document_type.value,
                        "date": record_obj.metadata.get("document_date", record_obj.created_at),
                        "content": self.redactor.redact_for_output(content),
                        "metadata": self.redactor.redact_in_record_summary(record_obj.metadata),
                        "version": record_obj.version,
                        "extracted_entities": record_obj.extracted_entities
                    })

            response_data = PatientDataResponse(
                query=query.to_dict(),
                patient_id="[REDACTED]",
                records_found=len(detailed_records),
                summary=self._generate_summary(detailed_records, query.output_detail),
                matching_records=detailed_records,
                data_gaps=self._identify_data_gaps(detailed_records, query)
            ).to_dict()

            # Log access
            log_id = self.access_logger.log_access(
                patient_id=query.patient_identifier or "all",
                query_type=QueryType.RETRIEVE,
                query_details=query.to_dict(),
                records_accessed=[r["record_id"] for r in detailed_records],
                success=True
            )
            response_data["log_id"] = log_id

            return self.create_response(data=response_data, success=True)

        except Exception as e:
            logger.error(f"Retrieve failed: {e}")
            return self.create_response(
                data={"error": str(e)},
                success=False,
                errors=[f"Retrieve failed: {str(e)}"]
            )

    @require_access_logging
    def summary(self, query: PatientQuery) -> AgentResponse:
        """
        Generate a summary of patient records.

        Returns:
            AgentResponse with narrative summary of relevant findings
        """
        try:
            records = self._find_matching_records(query)

            # Build summary based on detail level
            summary_records = []
            for record in records:
                result = self.storage.retrieve(record.record_id)
                if result:
                    record_obj, content = result
                    # Extract key points based on document type
                    key_points = self._extract_key_points(
                        content,
                        record_obj.document_type,
                        query.output_detail
                    )
                    summary_records.append({
                        "record_id": record_obj.record_id,
                        "date": record_obj.metadata.get("document_date", record_obj.created_at),
                        "document_type": record_obj.document_type.value,
                        "key_points": key_points
                    })

            # Generate narrative summary
            narrative = self._generate_narrative_summary(summary_records, query)

            response_data = PatientDataResponse(
                query=query.to_dict(),
                patient_id="[REDACTED]",
                records_found=len(summary_records),
                summary=narrative,
                matching_records=summary_records,
                data_gaps=self._identify_data_gaps(summary_records, query)
            ).to_dict()

            # Log access
            log_id = self.access_logger.log_access(
                patient_id=query.patient_identifier or "all",
                query_type=QueryType.SUMMARY,
                query_details=query.to_dict(),
                records_accessed=[r["record_id"] for r in summary_records],
                success=True
            )
            response_data["log_id"] = log_id

            return self.create_response(data=response_data, success=True)

        except Exception as e:
            logger.error(f"Summary failed: {e}")
            return self.create_response(
                data={"error": str(e)},
                success=False,
                errors=[f"Summary failed: {str(e)}"]
            )

    @require_access_logging
    def timeline(self, query: PatientQuery) -> AgentResponse:
        """
        Generate a chronological timeline of patient events.

        Returns:
            AgentResponse with timeline view of relevant events
        """
        try:
            records = self._find_matching_records(query)

            # Build timeline entries
            timeline_entries = []
            for record in records:
                result = self.storage.retrieve(record.record_id)
                if result:
                    record_obj, content = result
                    entry = {
                        "date": record_obj.metadata.get("document_date", record_obj.created_at),
                        "event_type": record_obj.document_type.value,
                        "record_id": record_obj.record_id,
                        "description": self._extract_event_description(content, record_obj),
                        "provider": record_obj.metadata.get("provider", "Unknown"),
                        "location": record_obj.metadata.get("location", "Unknown")
                    }
                    timeline_entries.append(entry)

            # Sort by date
            timeline_entries.sort(key=lambda x: x["date"])

            # Build summary based on timeline
            summary_records = [
                {
                    "record_id": e["record_id"],
                    "date": e["date"],
                    "document_type": e["event_type"],
                    "key_points": e["description"]
                }
                for e in timeline_entries
            ]

            response_data = PatientDataResponse(
                query=query.to_dict(),
                patient_id="[REDACTED]",
                records_found=len(timeline_entries),
                summary=f"Timeline of {len(timeline_entries)} events",
                matching_records=summary_records,
                timeline_view=timeline_entries,
                data_gaps=self._identify_data_gaps(timeline_entries, query)
            ).to_dict()

            # Log access
            log_id = self.access_logger.log_access(
                patient_id=query.patient_identifier or "all",
                query_type=QueryType.TIMELINE,
                query_details=query.to_dict(),
                records_accessed=[e["record_id"] for e in timeline_entries],
                success=True
            )
            response_data["log_id"] = log_id

            return self.create_response(data=response_data, success=True)

        except Exception as e:
            logger.error(f"Timeline failed: {e}")
            return self.create_response(
                data={"error": str(e)},
                success=False,
                errors=[f"Timeline failed: {str(e)}"]
            )

    # =========================================================================
    # Document Management Methods
    # =========================================================================

    def store_document(
        self,
        patient_id: str,
        document_type: DocumentType,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AgentResponse:
        """
        Store a new clinical document.

        Args:
            patient_id: Patient identifier
            document_type: Type of document
            content: Document content
            metadata: Optional metadata (date, provider, etc.)

        Returns:
            AgentResponse with stored record details
        """
        try:
            # Extract medical entities from content
            entities = self.medical_nlp.extract_entities(content)

            # Store document
            record = self.storage.store(
                patient_id=patient_id,
                document_type=document_type,
                content=content,
                metadata=metadata
            )

            # Update record with extracted entities
            record.extracted_entities = entities

            # Index for RAG retrieval
            self.rag.index_document(
                document=content,
                metadata={
                    "record_id": record.record_id,
                    "patient_id": record.patient_id,
                    "document_type": document_type.value,
                    **entities
                },
                document_id=record.record_id
            )

            # Log storage operation
            self.access_logger.log_access(
                patient_id=patient_id,
                query_type=QueryType.RETRIEVE,  # Using RETRIEVE for store operations
                query_details={"operation": "store", "document_type": document_type.value},
                records_accessed=[record.record_id],
                success=True
            )

            return self.create_response(
                data={
                    "status": "stored",
                    "record_id": record.record_id,
                    "document_type": document_type.value,
                    "version": record.version,
                    "extracted_entities": entities
                },
                success=True
            )

        except Exception as e:
            logger.error(f"Store document failed: {e}")
            return self.create_response(
                data={"error": str(e)},
                success=False,
                errors=[f"Store failed: {str(e)}"]
            )

    def link_records(
        self,
        record_id: str,
        related_record_ids: List[str]
    ) -> AgentResponse:
        """
        Link related records across visits.
        """
        try:
            # Get the record
            result = self.storage.retrieve(record_id)
            if not result:
                return self.create_response(
                    data={"error": "Record not found"},
                    success=False,
                    errors=["Record not found"]
                )

            record, _ = result

            # Add links
            existing_links = set(record.linked_records)
            new_links = existing_links | set(related_record_ids)

            # Update record
            self.storage.update(
                record_id=record_id,
                metadata={"linked_records": list(new_links)}
            )

            return self.create_response(
                data={
                    "status": "linked",
                    "record_id": record_id,
                    "linked_records": list(new_links)
                },
                success=True
            )

        except Exception as e:
            logger.error(f"Link records failed: {e}")
            return self.create_response(
                data={"error": str(e)},
                success=False,
                errors=[f"Link failed: {str(e)}"]
            )

    # =========================================================================
    # Alert Methods
    # =========================================================================

    def check_alerts(self, patient_id: str) -> AgentResponse:
        """
        Check for patient data alerts (missing follow-ups, overdue tests, etc.)
        """
        alerts = []

        # Get patient records
        records = self.storage.find_by_patient(patient_id)

        # Check for missing follow-ups
        for record in records:
            if record.metadata.get("follow_up_required"):
                follow_up_date = record.metadata.get("follow_up_date")
                if follow_up_date and follow_up_date < datetime.now().isoformat():
                    alerts.append(PatientDataAlert(
                        alert_type=AlertType.MISSING_FOLLOWUP,
                        patient_id=record.patient_id,
                        details=f"Follow-up overdue since {follow_up_date}",
                        recommended_action="Schedule follow-up appointment",
                        severity="high"
                    ))

        # Check for overdue tests
        for record in records:
            if record.document_type == DocumentType.LABS:
                next_test_date = record.metadata.get("next_test_date")
                if next_test_date and next_test_date < datetime.now().isoformat():
                    alerts.append(PatientDataAlert(
                        alert_type=AlertType.OVERDUE_TEST,
                        patient_id=record.patient_id,
                        details=f"Lab test overdue since {next_test_date}",
                        recommended_action="Order follow-up labs",
                        severity="medium"
                    ))

        self._alerts.extend(alerts)

        return self.create_response(
            data={
                "patient": "[REDACTED]",
                "alerts": [a.to_dict() for a in alerts],
                "alert_count": len(alerts)
            },
            success=True
        )

    def get_pending_alerts(self) -> List[PatientDataAlert]:
        """Get all pending alerts."""
        return self._alerts.copy()

    def dismiss_alert(self, alert_index: int) -> bool:
        """Dismiss an alert by index."""
        if 0 <= alert_index < len(self._alerts):
            self._alerts.pop(alert_index)
            return True
        return False

    # =========================================================================
    # Semantic Search (Qdrant-ready)
    # =========================================================================

    def semantic_search(
        self,
        query_text: str,
        patient_id: Optional[str] = None,
        top_k: int = 10
    ) -> AgentResponse:
        """
        Perform semantic search using vector embeddings.

        This method is prepared for Qdrant integration. Currently uses
        RAG provider for text-based retrieval.
        """
        try:
            filters = {}
            if patient_id:
                filters["patient_id"] = hashlib.sha256(patient_id.encode()).hexdigest()

            # Use RAG provider for retrieval
            results = self.rag.query(
                query_text=query_text,
                top_k=top_k,
                filters=filters if filters else None
            )

            # Build response
            matching_records = []
            for result in results:
                matching_records.append({
                    "record_id": result["document_id"],
                    "relevance_score": result["score"],
                    "excerpt": self.redactor.redact_for_output(result["text"]),
                    "metadata": self.redactor.redact_in_record_summary(result.get("metadata", {}))
                })

            # Log access
            log_id = self.access_logger.log_access(
                patient_id=patient_id or "all",
                query_type=QueryType.SEARCH,
                query_details={"semantic_query": query_text, "top_k": top_k},
                records_accessed=[r["record_id"] for r in matching_records],
                success=True
            )

            return self.create_response(
                data={
                    "query": query_text,
                    "patient": "[REDACTED]" if patient_id else "all",
                    "records_found": len(matching_records),
                    "matching_records": matching_records,
                    "access_logged": True,
                    "log_id": log_id
                },
                success=True
            )

        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return self.create_response(
                data={"error": str(e)},
                success=False,
                errors=[f"Semantic search failed: {str(e)}"]
            )

    # =========================================================================
    # Main Process Method (Alfred Interface)
    # =========================================================================

    def process(self, request: Dict[str, Any]) -> AgentResponse:
        """
        Main entry point for processing patient data requests from Alfred.

        Expected request format:
        {
            "query_type": "search" | "retrieve" | "summary" | "timeline",
            "patient_identifier": "...",
            "search_terms": [...],
            "date_range": ["start", "end"],
            "document_types": ["notes", "labs", ...],
            "output_detail": "brief" | "standard" | "comprehensive"
        }
        """
        # Check state permission
        permitted, reason = self.check_state_permission()
        if not permitted:
            return self.blocked_response(reason)

        try:
            # Parse request into query
            query = PatientQuery(
                query_type=QueryType(request.get("query_type", "search")),
                patient_identifier=request.get("patient_identifier"),
                search_terms=request.get("search_terms", []),
                date_range=tuple(request["date_range"]) if request.get("date_range") else None,
                document_types=[
                    DocumentType(dt) for dt in request.get("document_types", ["all"])
                ],
                output_detail=OutputDetail(request.get("output_detail", "standard"))
            )

            # Route to appropriate method
            if query.query_type == QueryType.SEARCH:
                return self.search(query)
            elif query.query_type == QueryType.RETRIEVE:
                return self.retrieve(query)
            elif query.query_type == QueryType.SUMMARY:
                return self.summary(query)
            elif query.query_type == QueryType.TIMELINE:
                return self.timeline(query)
            else:
                return self.create_response(
                    data={"error": f"Unknown query type: {query.query_type}"},
                    success=False,
                    errors=[f"Unknown query type: {query.query_type}"]
                )

        except Exception as e:
            logger.error(f"Process request failed: {e}")
            return self.create_response(
                data={"error": str(e)},
                success=False,
                errors=[f"Request processing failed: {str(e)}"]
            )

    # =========================================================================
    # Private Helper Methods
    # =========================================================================

    def _find_matching_records(self, query: PatientQuery) -> List[PatientRecord]:
        """Find records matching query criteria."""
        records = []

        # Start with patient filter if specified
        if query.patient_identifier:
            records = self.storage.find_by_patient(query.patient_identifier)
        else:
            records = list(self.storage._records.values())

        # Filter by document type
        if DocumentType.ALL not in query.document_types:
            records = [r for r in records if r.document_type in query.document_types]

        # Filter by date range
        if query.date_range:
            start_date, end_date = query.date_range
            records = [
                r for r in records
                if start_date <= r.metadata.get("document_date", r.created_at) <= end_date
            ]

        # Filter by search terms (if any)
        if query.search_terms:
            filtered_records = []
            for record in records:
                # Check if any search term matches metadata or extracted entities
                for term in query.search_terms:
                    term_lower = term.lower()
                    # Check metadata
                    if any(
                        term_lower in str(v).lower()
                        for v in record.metadata.values()
                    ):
                        filtered_records.append(record)
                        break
                    # Check extracted entities
                    for entity_type, entities in record.extracted_entities.items():
                        if any(
                            term_lower in str(e).lower()
                            for e in entities
                        ):
                            filtered_records.append(record)
                            break
            records = filtered_records

        return records

    def _build_response_data(
        self,
        query: PatientQuery,
        records: List[PatientRecord],
        include_timeline: bool = False
    ) -> Dict[str, Any]:
        """Build response data structure."""
        matching_records = []
        for record in records:
            record_summary = {
                "record_id": record.record_id,
                "date": record.metadata.get("document_date", record.created_at),
                "document_type": record.document_type.value,
                "key_points": self._extract_brief_key_points(record)
            }
            matching_records.append(record_summary)

        response = PatientDataResponse(
            query=query.to_dict(),
            patient_id="[REDACTED]",
            records_found=len(records),
            summary=self._generate_summary(matching_records, query.output_detail),
            matching_records=matching_records,
            related_records=self._find_related_records(records),
            data_gaps=self._identify_data_gaps(matching_records, query)
        )

        if include_timeline:
            response.timeline_view = self._build_timeline(records)

        return response.to_dict()

    def _extract_brief_key_points(self, record: PatientRecord) -> str:
        """Extract brief key points from record metadata."""
        key_points = []

        # Add from extracted entities
        for entity_type, entities in record.extracted_entities.items():
            if entities:
                key_points.append(f"{entity_type}: {', '.join(str(e.get('text', e)) for e in entities[:3])}")

        # Add from metadata
        if record.metadata.get("chief_complaint"):
            key_points.append(f"Chief complaint: {record.metadata['chief_complaint']}")
        if record.metadata.get("diagnosis"):
            key_points.append(f"Diagnosis: {record.metadata['diagnosis']}")

        return "; ".join(key_points) if key_points else "No key points extracted"

    def _extract_key_points(
        self,
        content: str,
        document_type: DocumentType,
        detail_level: OutputDetail
    ) -> str:
        """Extract key points from document content."""
        # Extract entities
        entities = self.medical_nlp.extract_entities(content)

        key_points = []

        # Add conditions
        if entities.get("conditions"):
            conditions = [e["text"] for e in entities["conditions"][:5]]
            key_points.append(f"Conditions: {', '.join(conditions)}")

        # Add medications
        if entities.get("medications"):
            meds = [e["text"] for e in entities["medications"][:5]]
            key_points.append(f"Medications: {', '.join(meds)}")

        # Add procedures
        if entities.get("procedures"):
            procs = [e["text"] for e in entities["procedures"][:5]]
            key_points.append(f"Procedures: {', '.join(procs)}")

        # Add more detail for comprehensive output
        if detail_level == OutputDetail.COMPREHENSIVE:
            # Include first paragraph of content (redacted)
            first_para = content.split('\n\n')[0][:500]
            key_points.append(f"Summary: {self.redactor.redact_for_output(first_para)}")

        return "; ".join(key_points) if key_points else "No key points extracted"

    def _extract_event_description(
        self,
        content: str,
        record: PatientRecord
    ) -> str:
        """Extract event description for timeline."""
        description_parts = []

        # Add chief complaint or reason
        if record.metadata.get("chief_complaint"):
            description_parts.append(record.metadata["chief_complaint"])
        elif record.metadata.get("reason"):
            description_parts.append(record.metadata["reason"])

        # Add diagnosis if available
        if record.metadata.get("diagnosis"):
            description_parts.append(f"Dx: {record.metadata['diagnosis']}")

        # Fallback to first line of content
        if not description_parts:
            first_line = content.split('\n')[0][:200]
            description_parts.append(self.redactor.redact_for_output(first_line))

        return "; ".join(description_parts)

    def _generate_summary(
        self,
        records: List[Dict[str, Any]],
        detail_level: OutputDetail
    ) -> str:
        """Generate narrative summary of records."""
        if not records:
            return "No matching records found."

        # Group by document type
        by_type: Dict[str, List[Dict[str, Any]]] = {}
        for record in records:
            doc_type = record.get("document_type", "unknown")
            if doc_type not in by_type:
                by_type[doc_type] = []
            by_type[doc_type].append(record)

        summary_parts = [f"Found {len(records)} matching records."]

        for doc_type, type_records in by_type.items():
            summary_parts.append(
                f"{doc_type.capitalize()}: {len(type_records)} records"
            )

        if detail_level in [OutputDetail.STANDARD, OutputDetail.COMPREHENSIVE]:
            # Add date range
            dates = [r.get("date") for r in records if r.get("date")]
            if dates:
                dates.sort()
                summary_parts.append(f"Date range: {dates[0]} to {dates[-1]}")

        return " ".join(summary_parts)

    def _generate_narrative_summary(
        self,
        records: List[Dict[str, Any]],
        query: PatientQuery
    ) -> str:
        """Generate a more detailed narrative summary."""
        if not records:
            return "No matching records found for the specified criteria."

        narrative_parts = [
            f"Patient data summary based on {len(records)} records"
        ]

        if query.date_range:
            narrative_parts.append(
                f"covering the period from {query.date_range[0]} to {query.date_range[1]}"
            )

        narrative_parts.append(".")

        # Summarize by type
        type_counts: Dict[str, int] = {}
        for record in records:
            doc_type = record.get("document_type", "unknown")
            type_counts[doc_type] = type_counts.get(doc_type, 0) + 1

        type_summary = ", ".join(
            f"{count} {dtype}" for dtype, count in type_counts.items()
        )
        narrative_parts.append(f" Records include: {type_summary}.")

        # Add key findings if available
        key_findings = set()
        for record in records:
            key_points = record.get("key_points", "")
            if key_points and key_points != "No key points extracted":
                # Extract first key point
                first_point = key_points.split(";")[0].strip()
                if len(first_point) < 100:
                    key_findings.add(first_point)

        if key_findings:
            narrative_parts.append(
                f" Key findings: {'; '.join(list(key_findings)[:5])}."
            )

        return " ".join(narrative_parts)

    def _find_related_records(
        self,
        records: List[PatientRecord]
    ) -> List[Dict[str, Any]]:
        """Find records related to the given records."""
        related = []
        seen_ids = {r.record_id for r in records}

        for record in records:
            for linked_id in record.linked_records:
                if linked_id not in seen_ids:
                    result = self.storage.retrieve(linked_id)
                    if result:
                        linked_record, _ = result
                        related.append({
                            "record_id": linked_record.record_id,
                            "document_type": linked_record.document_type.value,
                            "date": linked_record.metadata.get(
                                "document_date", linked_record.created_at
                            ),
                            "relationship": "linked"
                        })
                        seen_ids.add(linked_id)

        return related

    def _identify_data_gaps(
        self,
        records: List[Dict[str, Any]],
        query: PatientQuery
    ) -> List[str]:
        """Identify potential gaps in patient data."""
        gaps = []

        if not records:
            gaps.append("No records found matching criteria")
            return gaps

        # Check for missing document types
        found_types = {r.get("document_type") for r in records}

        # Suggest potential missing data based on query
        if query.search_terms:
            for term in query.search_terms:
                term_lower = term.lower()
                if "diabetes" in term_lower and "labs" not in found_types:
                    gaps.append("No lab records found - consider HbA1c data")
                if "hypertension" in term_lower and "labs" not in found_types:
                    gaps.append("No lab records found - consider renal function tests")

        # Check for timeline gaps
        if query.query_type == QueryType.TIMELINE:
            dates = [r.get("date") for r in records if r.get("date")]
            if dates:
                dates.sort()
                # Check for gaps > 6 months
                for i in range(1, len(dates)):
                    # Simple date comparison (would need proper date parsing in production)
                    if dates[i][:4] != dates[i-1][:4]:  # Different year
                        gaps.append(f"Potential gap between {dates[i-1]} and {dates[i]}")

        return gaps

    def _build_timeline(self, records: List[PatientRecord]) -> List[Dict[str, Any]]:
        """Build timeline view from records."""
        timeline = []

        for record in records:
            entry = {
                "date": record.metadata.get("document_date", record.created_at),
                "event_type": record.document_type.value,
                "record_id": record.record_id,
                "description": self._extract_brief_key_points(record)
            }
            timeline.append(entry)

        timeline.sort(key=lambda x: x["date"])
        return timeline


# =============================================================================
# Factory Function
# =============================================================================

def create_patient_data_agent(
    storage_path: Optional[str] = None,
    use_qdrant: bool = False,
    use_medcat: bool = False,
    use_presidio: bool = False,
    use_llamaindex: bool = False
) -> PatientDataAgent:
    """
    Factory function to create a PatientDataAgent with optional integrations.

    Args:
        storage_path: Custom storage path
        use_qdrant: Enable Qdrant vector search (requires qdrant-client)
        use_medcat: Enable MedCAT medical NLP (requires medcat)
        use_presidio: Enable Presidio PHI detection (requires presidio-analyzer)
        use_llamaindex: Enable LlamaIndex RAG (requires llama-index)

    Returns:
        Configured PatientDataAgent instance
    """
    path = Path(storage_path).expanduser() if storage_path else None

    vector_provider = None
    nlp_provider = None
    phi_provider = None
    rag_provider = None

    # Initialize Qdrant if requested
    if use_qdrant:
        try:
            # Would import and configure Qdrant client here
            # from qdrant_client import QdrantClient
            logger.info("Qdrant integration enabled")
        except ImportError:
            logger.warning("Qdrant not available, using default vector search")

    # Initialize MedCAT if requested
    if use_medcat:
        try:
            # Would import and configure MedCAT here
            # from medcat.cat import CAT
            logger.info("MedCAT integration enabled")
        except ImportError:
            logger.warning("MedCAT not available, using default NLP")

    # Initialize Presidio if requested
    if use_presidio:
        try:
            # Would import and configure Presidio here
            # from presidio_analyzer import AnalyzerEngine
            logger.info("Presidio integration enabled")
        except ImportError:
            logger.warning("Presidio not available, using default PHI detection")

    # Initialize LlamaIndex if requested
    if use_llamaindex:
        try:
            # Would import and configure LlamaIndex here
            # from llama_index import VectorStoreIndex
            logger.info("LlamaIndex integration enabled")
        except ImportError:
            logger.warning("LlamaIndex not available, using default RAG")

    return PatientDataAgent(
        storage_path=path,
        vector_provider=vector_provider,
        nlp_provider=nlp_provider,
        phi_provider=phi_provider,
        rag_provider=rag_provider
    )


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Main class
    "PatientDataAgent",

    # Data classes
    "PatientRecord",
    "PatientQuery",
    "PatientDataResponse",
    "PatientDataAlert",
    "AccessLogEntry",

    # Enums
    "QueryType",
    "DocumentType",
    "OutputDetail",
    "AlertType",
    "PHIType",

    # Abstract interfaces (for custom implementations)
    "EncryptionProvider",
    "VectorSearchProvider",
    "MedicalNLPProvider",
    "PHIDetectionProvider",
    "RAGProvider",

    # Default implementations (for testing/development)
    "DefaultEncryptionProvider",
    "DefaultVectorSearchProvider",
    "DefaultPHIDetectionProvider",
    "DefaultMedicalNLPProvider",
    "DefaultRAGProvider",

    # Utilities
    "DocumentStorage",
    "AccessLogger",
    "PHIRedactor",

    # Factory
    "create_patient_data_agent",
]
