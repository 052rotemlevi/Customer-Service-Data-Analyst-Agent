"""DynamoDB-based checkpointer for AWS Lambda deployment.

Implements the LangGraph BaseCheckpointSaver interface using DynamoDB
for serverless-compatible persistent conversation memory.
"""

import json
import time
from typing import Any, Iterator, Optional, Sequence, Tuple

import boto3
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
)


class DynamoDBSaver(BaseCheckpointSaver):
    """DynamoDB-backed checkpoint saver for LangGraph.

    Stores conversation state in a DynamoDB table with session_id as
    partition key and checkpoint_id as sort key.
    """

    def __init__(self, table_name: str, region_name: str | None = None):
        super().__init__()
        self.dynamodb = boto3.resource("dynamodb", region_name=region_name)
        self.table = self.dynamodb.Table(table_name)

    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """Get the latest checkpoint for a thread."""
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")

        # Check if a specific checkpoint_id is requested
        checkpoint_id = config["configurable"].get("checkpoint_id")

        if checkpoint_id:
            response = self.table.get_item(
                Key={"session_id": f"{thread_id}:{checkpoint_ns}", "checkpoint_id": checkpoint_id}
            )
            item = response.get("Item")
            if not item:
                return None
        else:
            # Get the latest checkpoint
            response = self.table.query(
                KeyConditionExpression=boto3.dynamodb.conditions.Key("session_id").eq(
                    f"{thread_id}:{checkpoint_ns}"
                ),
                ScanIndexForward=False,
                Limit=1,
            )
            items = response.get("Items", [])
            if not items:
                return None
            item = items[0]

        checkpoint = json.loads(item["checkpoint_data"])
        metadata = json.loads(item.get("metadata", "{}"))

        return CheckpointTuple(
            config=config,
            checkpoint=checkpoint,
            metadata=metadata,
            parent_config=None,
        )

    def list(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> Iterator[CheckpointTuple]:
        """List checkpoints for a thread."""
        if not config:
            return

        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")

        response = self.table.query(
            KeyConditionExpression=boto3.dynamodb.conditions.Key("session_id").eq(
                f"{thread_id}:{checkpoint_ns}"
            ),
            ScanIndexForward=False,
            Limit=limit or 10,
        )

        for item in response.get("Items", []):
            checkpoint = json.loads(item["checkpoint_data"])
            metadata = json.loads(item.get("metadata", "{}"))
            yield CheckpointTuple(
                config=config,
                checkpoint=checkpoint,
                metadata=metadata,
                parent_config=None,
            )

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: Optional[dict[str, Any]] = None,
    ) -> RunnableConfig:
        """Save a checkpoint."""
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = checkpoint["id"]

        self.table.put_item(
            Item={
                "session_id": f"{thread_id}:{checkpoint_ns}",
                "checkpoint_id": checkpoint_id,
                "checkpoint_data": json.dumps(checkpoint, default=str),
                "metadata": json.dumps(metadata, default=str),
                "timestamp": int(time.time()),
            }
        )

        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        }

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[Tuple[str, Any]],
        task_id: str,
    ) -> None:
        """Store intermediate writes (not critical for basic operation)."""
        pass
