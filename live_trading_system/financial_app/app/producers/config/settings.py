"""
Configuration settings for Kinesis producers.
"""

import os
from typing import Optional

class KinesisSettings:
    """Settings for Kinesis configuration."""
    
    def __init__(
        self,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_region: Optional[str] = None
    ):
        """
        Initialize Kinesis settings.
        
        Args:
            aws_access_key_id: AWS access key ID
            aws_secret_access_key: AWS secret access key
            aws_region: AWS region name
        """
        self.AWS_ACCESS_KEY_ID = aws_access_key_id or os.getenv('AWS_ACCESS_KEY_ID')
        self.AWS_SECRET_ACCESS_KEY = aws_secret_access_key or os.getenv('AWS_SECRET_ACCESS_KEY')
        self.AWS_REGION = aws_region or os.getenv('AWS_REGION', 'us-east-1') 