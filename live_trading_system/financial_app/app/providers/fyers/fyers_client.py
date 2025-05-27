"""
Fyers API client implementation.

This module provides a low-level client for interacting with the Fyers API,
handling authentication, market data requests, and WebSocket connections.
"""

import yaml
import logging
import requests
import json
import time
import hashlib
import urllib.parse
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class FyersClient:
    def __init__(self, config_path: str = "config/broker_config.yaml"):
        self.config = self._load_config(config_path)
        self.broker_config = self._get_broker_config()
        self.session = requests.Session()
        self.access_token = None
        self.base_url = self.broker_config["base_url"]
        self.ws_url = self.broker_config["ws_url"]
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        try:
            with open(config_path, "r") as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load broker configuration: {e}")
            raise
               
    def _get_broker_config(self) -> Dict[str, Any]:
        broker_config = self.config.get("brokers", {}).get("fyers")
        if not broker_config:
            raise ValueError("Fyers broker not found in configuration")
        if not broker_config.get("enabled", False):
            raise ValueError("Fyers broker is disabled in configuration")
        return broker_config
    
    def connect(self) -> bool:
        """Establish connection to Fyers API"""
        try:
            # Generate access token
            success = self._generate_access_token()
            if not success:
                return False
                
            # Test connection with a simple API call
            profile = self.get_profile()
            if profile:
                logger.info(f"Successfully connected to Fyers API for user: {profile.get('name')}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to connect to Fyers API: {e}")
            return False
    
    def _generate_access_token(self) -> bool:
        """Generate access token using the Fyers API authentication flow"""
        try:
            # Step 1: Generate auth code
            auth_code = self._get_auth_code()
            if not auth_code:
                return False
                
            # Step 2: Exchange auth code for access token
            # The token URL from the documentation
            token_url = f"{self.base_url}/validate-authcode"
            
            # Prepare the payload according to the documentation
            payload = {
                "grant_type": "authorization_code",
                "appIdHash": self._generate_app_id_hash(),
                "code": auth_code
            }
            
            # Set the headers properly
            headers = {
                "Content-Type": "application/json"
            }
            
            logger.info(f"Attempting to exchange auth code for access token")
            logger.info(f"Token URL: {token_url}")
            
            response = self.session.post(token_url, json=payload, headers=headers)
            
            # Log the response for debugging
            logger.info(f"Token response status: {response.status_code}")
            logger.info(f"Token response: {response.text}")
            
            if response.status_code != 200:
                logger.error(f"Failed to get access token: {response.text}")
                return False
                
            token_data = response.json()
            if not token_data.get("access_token"):
                logger.error(f"No access token in response: {token_data}")
                return False
                
            self.access_token = token_data["access_token"]
            # Set authorization header for future requests
            self.session.headers.update({
                "Authorization": f"{self.broker_config['api_id']}:{self.access_token}"
            })
            return True
        except Exception as e:
            logger.error(f"Error generating access token: {e}", exc_info=True)
            return False
        
    def _get_auth_code(self) -> Optional[str]:
        """
        Get authorization code
        
        Note: This is a simplified version. In a real application, you'd need to:
        1. Redirect user to Fyers login page
        2. Handle the callback with the auth code
        
        For testing, you might need to manually get this code and input it
        """
        # For testing purposes, you might want to manually input the auth code
        # or implement a proper OAuth flow with a local web server to catch the callback
        
        # Example URL to get auth code (user needs to login and approve)
        app_id = self.broker_config["api_id"]
        redirect_uri = urllib.parse.quote(self.broker_config["redirect_uri"])
        
        # Use the correct base URL (api-t1.fyers.in for testing)
        auth_url = f"https://api-t1.fyers.in/api/v3/generate-authcode?client_id={app_id}&redirect_uri={redirect_uri}&response_type=code&state=None"
        
        print("Please visit the following URL to authenticate:")
        print(auth_url)
        print("After authentication, enter the authorization code from the redirect URL:")
        print("(Look for the 'code=' parameter in the redirected URL)")
        auth_code = input("Authorization code: ")
        return auth_code.strip() if auth_code else None
    
    def _generate_app_id_hash(self) -> str:
        """Generate app ID hash required for token generation"""
        app_id = self.broker_config["api_id"]
        app_secret = self.broker_config["api_secret"]
        data = f"{app_id}:{app_secret}"
        return hashlib.sha256(data.encode()).hexdigest()
    
    def get_profile(self) -> Dict[str, Any]:
        """Get user profile to verify connection"""
        if not self.access_token:
            logger.error("Not authenticated. Call connect() first.")
            return {}
            
        try:
            profile_url = f"{self.base_url}/profile"
            response = self.session.get(profile_url)
            if response.status_code != 200:
                logger.error(f"Failed to get profile: {response.text}")
                return {}
                
            profile_data = response.json()
            return profile_data.get("data", {})
        except Exception as e:
            logger.error(f"Error getting profile: {e}")
            return {}
    
    def get_market_data(self, symbols: List[str]) -> Dict[str, Any]:
        """Get market data for specified symbols"""
        if not self.access_token:
            logger.error("Not authenticated. Call connect() first.")
            return {}
            
        try:
            quotes_url = f"{self.base_url}/quotes"
            symbols_str = ",".join(symbols)
            params = {"symbols": symbols_str}
            
            response = self.session.get(quotes_url, params=params)
            if response.status_code != 200:
                logger.error(f"Failed to get market data: {response.text}")
                return {}
                
            market_data = response.json()
            return market_data.get("data", {})
        except Exception as e:
            logger.error(f"Error getting market data: {e}")
            return {}
    
    def subscribe_to_market_data(self, symbols: List[str]) -> bool:
        """
        Subscribe to real-time market data for specified symbols
        
        Note: This is a placeholder. You would need to implement
        WebSocket connection handling for real-time data subscription.
        """
        # In a real implementation, you would:
        # 1. Establish WebSocket connection
        # 2. Send subscription message with symbols
        # 3. Handle incoming data and connection management
        
        logger.info(f"Subscribing to market data for symbols: {symbols}")
        return True  # Placeholder 