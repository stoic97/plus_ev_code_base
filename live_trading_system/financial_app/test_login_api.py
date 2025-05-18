import requests

def test_login_debug():
    """Quick debug test for login"""
    url = "http://localhost:8000/api/v1/auth/login"
    data = {
        "username": "stoic125",
        "password": "123@Ev456#789"  # Make sure this is the exact password
    }
    
    print("Testing login...")
    response = requests.post(
        url,
        data=data,
        headers={"Content-Type": "application/x-www-form-urlencoded"}
    )
    
    print(f"Status: {response.status_code}")
    print(f"Response: {response.text}")

if __name__ == "__main__":
    test_login_debug()