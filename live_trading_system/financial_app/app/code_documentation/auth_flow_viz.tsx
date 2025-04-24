import React from 'react';
import { ChevronRight, Lock, UserPlus, LogIn, RefreshCw, Shield, Database, Server, FileCode, FileJson, User } from 'lucide-react';

const AuthFlowDiagram = () => {
  // Common styles
  const boxStyle = "p-3 rounded-lg border shadow-sm m-2 text-center";
  const arrowStyle = "flex items-center justify-center text-gray-500 px-2";
  const sectionStyle = "mb-8 border rounded-lg p-4 bg-gray-50";
  const headerStyle = "text-lg font-bold mb-4 text-blue-800";
  const subHeaderStyle = "text-md font-semibold mb-2 text-blue-700";
  const codeBoxStyle = "p-2 bg-gray-100 rounded border text-sm font-mono text-left";
  const stepStyle = "flex items-start my-3";
  const stepNumberStyle = "w-6 h-6 rounded-full bg-blue-500 text-white flex items-center justify-center mr-2 flex-shrink-0";
  const stepTextStyle = "text-sm";
  const fileReferenceStyle = "text-xs text-gray-500 italic mt-1";
  const moduleBoxStyle = `${boxStyle} border-blue-200 bg-blue-50`;
  const dataBoxStyle = `${boxStyle} border-green-200 bg-green-50`;
  const actionBoxStyle = `${boxStyle} border-orange-200 bg-orange-50`;
  const flowDividerStyle = "my-6 border-t border-gray-300 relative";
  const flowDividerTextStyle = "absolute top-0 left-1/2 transform -translate-x-1/2 -translate-y-1/2 bg-white px-4 text-gray-500 text-sm";

  return (
    <div className="p-4 w-full">
      <h1 className="text-2xl font-bold mb-6 text-center text-blue-900">Authentication Flow</h1>
      <p className="text-sm text-gray-600 mb-6 text-center">
        Trading Strategies Application Authentication System
      </p>

      {/* Architecture Overview */}
      <div className={sectionStyle}>
        <h2 className={headerStyle}>Authentication Architecture</h2>
        <div className="flex flex-wrap justify-center">
          <div className={`${moduleBoxStyle} w-64`}>
            <div className="flex justify-center mb-2">
              <FileCode className="text-blue-700" size={24} />
            </div>
            <div className="font-semibold mb-1">API Endpoints</div>
            <div className="text-xs text-gray-600">app/api/v1/endpoints/auth.py</div>
            <div className="text-xs mt-2">Registration, Login, Token Refresh, Password Reset</div>
          </div>
          
          <div className={arrowStyle}>
            <ChevronRight size={20} />
          </div>
          
          <div className={`${moduleBoxStyle} w-64`}>
            <div className="flex justify-center mb-2">
              <Lock className="text-blue-700" size={24} />
            </div>
            <div className="font-semibold mb-1">Security Core</div>
            <div className="text-xs text-gray-600">app/core/security.py</div>
            <div className="text-xs mt-2">JWT Generation, Password Hashing, User Authentication</div>
          </div>
          
          <div className={arrowStyle}>
            <ChevronRight size={20} />
          </div>
          
          <div className={`${moduleBoxStyle} w-64`}>
            <div className="flex justify-center mb-2">
              <Database className="text-blue-700" size={24} />
            </div>
            <div className="font-semibold mb-1">Database Models</div>
            <div className="text-xs text-gray-600">app/models/users.py</div>
            <div className="text-xs mt-2">User, Role, Session, APIKey Models</div>
          </div>
        </div>

        <div className="flex justify-center mt-4">
          <div className={`${moduleBoxStyle} w-64`}>
            <div className="flex justify-center mb-2">
              <Shield className="text-blue-700" size={24} />
            </div>
            <div className="font-semibold mb-1">Auth Middleware</div>
            <div className="text-xs text-gray-600">app/middleware/auth.py</div>
            <div className="text-xs mt-2">Route Protection, Authentication Context</div>
          </div>
          
          <div className={arrowStyle}>
            <ChevronRight size={20} />
          </div>
          
          <div className={`${moduleBoxStyle} w-64`}>
            <div className="flex justify-center mb-2">
              <FileJson className="text-blue-700" size={24} />
            </div>
            <div className="font-semibold mb-1">Validation Schemas</div>
            <div className="text-xs text-gray-600">app/schemas/auth.py</div>
            <div className="text-xs mt-2">Request/Response Data Validation</div>
          </div>
        </div>

        <div className="mt-4 p-2 bg-yellow-50 border border-yellow-100 rounded text-sm text-yellow-800">
          <div className="font-semibold">Note: Missing Service Layer</div>
          <div className="text-xs mt-1">
            The auth service layer (app/services/auth.py) is not implemented. Authentication logic is 
            handled directly in core/security.py and called from the endpoints.
          </div>
        </div>
      </div>

      {/* User Registration Flow */}
      <div className={sectionStyle}>
        <h2 className={headerStyle}>
          <UserPlus size={20} className="inline mr-2" />
          User Registration Flow
        </h2>
        
        <div className="flex flex-col">
          <div className={stepStyle}>
            <div className={stepNumberStyle}>1</div>
            <div className={stepTextStyle}>
              Client submits username, email, and password to <code>/api/v1/auth/register</code>
              <div className={fileReferenceStyle}>Handled by: app/api/v1/endpoints/auth.py::register()</div>
            </div>
          </div>
          
          <div className={stepStyle}>
            <div className={stepNumberStyle}>2</div>
            <div className={stepTextStyle}>
              Request data is validated using RegisterRequest schema
              <div className={fileReferenceStyle}>Using: app/schemas/auth.py::RegisterRequest</div>
              <div className={codeBoxStyle}>
                {`username: str = Field(..., min_length=3, max_length=50)
email: EmailStr
password: str = Field(..., min_length=12)
full_name: Optional[str] = None`}
              </div>
            </div>
          </div>
          
          <div className={stepStyle}>
            <div className={stepNumberStyle}>3</div>
            <div className={stepTextStyle}>
              Password strength is verified with complex requirements
              <div className={fileReferenceStyle}>Using: app/api/v1/endpoints/auth.py::verify_password_strength()</div>
            </div>
          </div>
          
          <div className={stepStyle}>
            <div className={stepNumberStyle}>4</div>
            <div className={stepTextStyle}>
              User is created with OBSERVER role
              <div className={fileReferenceStyle}>Using: app/core/security.py::create_user()</div>
            </div>
          </div>
          
          <div className={stepStyle}>
            <div className={stepNumberStyle}>5</div>
            <div className={stepTextStyle}>
              Password is hashed using bcrypt before storage
              <div className={fileReferenceStyle}>Using: app/core/security.py::get_password_hash()</div>
            </div>
          </div>
          
          <div className={stepStyle}>
            <div className={stepNumberStyle}>6</div>
            <div className={stepTextStyle}>
              Success response is returned with RegisterResponse schema
              <div className={fileReferenceStyle}>Using: app/schemas/auth.py::RegisterResponse</div>
            </div>
          </div>
        </div>
      </div>

      {/* User Login Flow */}
      <div className={sectionStyle}>
        <h2 className={headerStyle}>
          <LogIn size={20} className="inline mr-2" />
          User Login Flow
        </h2>
        
        <div className="flex flex-col">
          <div className={stepStyle}>
            <div className={stepNumberStyle}>1</div>
            <div className={stepTextStyle}>
              Client submits username and password to <code>/api/v1/auth/token</code> or <code>/api/v1/auth/login</code>
              <div className={fileReferenceStyle}>Handled by: app/api/v1/endpoints/auth.py::login_for_access_token() or login()</div>
            </div>
          </div>
          
          <div className={stepStyle}>
            <div className={stepNumberStyle}>2</div>
            <div className={stepTextStyle}>
              User credentials are authenticated
              <div className={fileReferenceStyle}>Using: app/core/security.py::authenticate_user()</div>
              <div className="mt-2 mb-2 ml-2 p-2 border-l-2 border-blue-300 pl-2 text-xs">
                <div className="font-semibold">Authentication Process:</div>
                <ol className="list-decimal ml-4 space-y-1">
                  <li>Get user from database by username</li>
                  <li>Verify password against stored hash using bcrypt</li>
                  <li>Check if user account is active (not disabled)</li>
                </ol>
              </div>
            </div>
          </div>
          
          <div className={stepStyle}>
            <div className={stepNumberStyle}>3</div>
            <div className={stepTextStyle}>
              JWT tokens are generated
              <div className={fileReferenceStyle}>Using: app/core/security.py::create_access_token() and create_refresh_token()</div>
              <div className="mt-2 mb-2 ml-2 p-2 border-l-2 border-green-300 pl-2 text-xs">
                <div className="font-semibold">Token Contents:</div>
                <ul className="list-disc ml-4 space-y-1">
                  <li><strong>Subject (sub)</strong>: Username</li>
                  <li><strong>Roles</strong>: User's assigned roles</li>
                  <li><strong>Expiration (exp)</strong>: Token expiry timestamp</li>
                </ul>
              </div>
            </div>
          </div>
          
          <div className={stepStyle}>
            <div className={stepNumberStyle}>4</div>
            <div className={stepTextStyle}>
              Login event is logged to audit trail
              <div className={fileReferenceStyle}>Using: app/core/security.py::log_auth_event()</div>
            </div>
          </div>
          
          <div className={stepStyle}>
            <div className={stepNumberStyle}>5</div>
            <div className={stepTextStyle}>
              Tokens are returned in response
              <div className={fileReferenceStyle}>Using: app/schemas/auth.py::TokenResponse or LoginResponse</div>
            </div>
          </div>
        </div>

        <div className="mt-4 p-2 border border-dashed rounded">
          <h3 className={subHeaderStyle}>JWT Token Lifecycle</h3>
          <div className="flex justify-between text-xs">
            <div className="text-center p-2 bg-blue-50 rounded border border-blue-100 w-64">
              <div className="font-semibold">Access Token</div>
              <div className="mt-1">Short-lived (30 minutes)</div>
              <div className="mt-1">Used for API authorization</div>
            </div>
            <div className="text-center p-2 bg-purple-50 rounded border border-purple-100 w-64">
              <div className="font-semibold">Refresh Token</div>
              <div className="mt-1">Long-lived (7 days)</div>
              <div className="mt-1">Used to get new access tokens</div>
            </div>
          </div>
        </div>
      </div>

      {/* Token Refresh Flow */}
      <div className={sectionStyle}>
        <h2 className={headerStyle}>
          <RefreshCw size={20} className="inline mr-2" />
          Token Refresh Flow
        </h2>
        
        <div className="flex flex-col">
          <div className={stepStyle}>
            <div className={stepNumberStyle}>1</div>
            <div className={stepTextStyle}>
              Client submits refresh token to <code>/api/v1/auth/refresh</code>
              <div className={fileReferenceStyle}>Handled by: app/api/v1/endpoints/auth.py::refresh_token()</div>
            </div>
          </div>
          
          <div className={stepStyle}>
            <div className={stepNumberStyle}>2</div>
            <div className={stepTextStyle}>
              Token is validated for:
              <ul className="list-disc ml-4 text-xs mt-1">
                <li>Cryptographic signature validity</li>
                <li>Expiration date</li>
                <li>User existence and active status</li>
              </ul>
              <div className={fileReferenceStyle}>Using: app/core/security.py::refresh_access_token()</div>
            </div>
          </div>
          
          <div className={stepStyle}>
            <div className={stepNumberStyle}>3</div>
            <div className={stepTextStyle}>
              New access and refresh tokens are generated
              <div className={fileReferenceStyle}>Using: app/core/security.py::create_access_token() and create_refresh_token()</div>
            </div>
          </div>
          
          <div className={stepStyle}>
            <div className={stepNumberStyle}>4</div>
            <div className={stepTextStyle}>
              Token refresh is logged to audit trail
              <div className={fileReferenceStyle}>Using: app/core/security.py::log_auth_event()</div>
            </div>
          </div>
          
          <div className={stepStyle}>
            <div className={stepNumberStyle}>5</div>
            <div className={stepTextStyle}>
              New tokens are returned in response
              <div className={fileReferenceStyle}>Using: app/schemas/auth.py::TokenResponse</div>
            </div>
          </div>
        </div>
      </div>

      {/* Protected Resource Access Flow */}
      <div className={sectionStyle}>
        <h2 className={headerStyle}>
          <Shield size={20} className="inline mr-2" />
          Protected Resource Access Flow
        </h2>
        
        <div className="flex flex-col">
          <div className={stepStyle}>
            <div className={stepNumberStyle}>1</div>
            <div className={stepTextStyle}>
              Client makes request to protected endpoint with Authorization header containing JWT token
              <div className="text-xs bg-gray-100 p-1 rounded mt-1 font-mono">
                Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
              </div>
            </div>
          </div>
          
          <div className={stepStyle}>
            <div className={stepNumberStyle}>2</div>
            <div className={stepTextStyle}>
              Auth middleware checks if path requires authentication
              <div className={fileReferenceStyle}>Using: app/middleware/auth.py::AuthMiddleware.dispatch()</div>
              <div className="mt-2 mb-2 ml-2 p-2 border-l-2 border-orange-300 pl-2 text-xs">
                <div className="font-semibold">Path Protection Logic:</div>
                <ul className="list-disc ml-4 space-y-1">
                  <li>Public paths like /docs, /auth/token are skipped</li>
                  <li>All other /api/v1/ paths require authentication</li>
                  <li>Middleware checks for presence of Authorization header</li>
                </ul>
              </div>
            </div>
          </div>
          
          <div className={stepStyle}>
            <div className={stepNumberStyle}>3</div>
            <div className={stepTextStyle}>
              Endpoint function uses dependency injection to verify token and get user
              <div className={fileReferenceStyle}>Using: app/core/security.py::get_current_user() or get_current_active_user()</div>
              <div className="mt-2 mb-2 ml-2 p-2 border-l-2 border-blue-300 pl-2 text-xs">
                <div className="font-semibold">Token Validation Process:</div>
                <ol className="list-decimal ml-4 space-y-1">
                  <li>JWT token is decoded and verified</li>
                  <li>User is retrieved from database</li>
                  <li>User active status is checked</li>
                  <li>Authentication is logged to audit trail</li>
                </ol>
              </div>
            </div>
          </div>
          
          <div className={stepStyle}>
            <div className={stepNumberStyle}>4</div>
            <div className={stepTextStyle}>
              For role-protected endpoints, role membership is verified
              <div className={fileReferenceStyle}>Using: app/core/security.py::has_role()</div>
              <div className={codeBoxStyle}>
                @router.get("/admin-only")
                async def admin_endpoint(
                    current_user: User = Depends(has_role([Roles.ADMIN]))
                ):
                    # Only admins can access this endpoint
              </div>
            </div>
          </div>
          
          <div className={stepStyle}>
            <div className={stepNumberStyle}>5</div>
            <div className={stepTextStyle}>
              Endpoint executes business logic and returns response
            </div>
          </div>
        </div>
      </div>

      {/* Database Schema */}
      <div className={sectionStyle}>
        <h2 className={headerStyle}>
          <Database size={20} className="inline mr-2" />
          Authentication Database Schema
        </h2>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="p-3 border rounded bg-white">
            <h3 className="font-semibold text-blue-900 mb-2">User</h3>
            <ul className="text-xs space-y-1">
              <li><strong>id</strong>: UUID (PK)</li>
              <li><strong>username</strong>: String (unique)</li>
              <li><strong>email</strong>: String (unique)</li>
              <li><strong>hashed_password</strong>: String</li>
              <li><strong>full_name</strong>: String</li>
              <li><strong>is_active</strong>: Boolean</li>
              <li><strong>is_superuser</strong>: Boolean</li>
              <li><strong>created_at</strong>: DateTime</li>
              <li><strong>last_login_at</strong>: DateTime</li>
              <li><strong>failed_login_attempts</strong>: Integer</li>
            </ul>
          </div>
          
          <div className="p-3 border rounded bg-white">
            <h3 className="font-semibold text-blue-900 mb-2">Role</h3>
            <ul className="text-xs space-y-1">
              <li><strong>id</strong>: Integer (PK)</li>
              <li><strong>name</strong>: String (unique)</li>
              <li><strong>description</strong>: String</li>
              <li><strong>permissions</strong>: JSON</li>
            </ul>
            <div className="text-xs mt-2 text-gray-600">
              Many-to-many relationship with User via user_roles table
            </div>
          </div>
          
          <div className="p-3 border rounded bg-white">
            <h3 className="font-semibold text-blue-900 mb-2">UserSession</h3>
            <ul className="text-xs space-y-1">
              <li><strong>id</strong>: UUID (PK)</li>
              <li><strong>user_id</strong>: UUID (FK)</li>
              <li><strong>token_id</strong>: String</li>
              <li><strong>ip_address</strong>: String</li>
              <li><strong>created_at</strong>: DateTime</li>
              <li><strong>expires_at</strong>: DateTime</li>
              <li><strong>is_revoked</strong>: Boolean</li>
            </ul>
          </div>
          
          <div className="p-3 border rounded bg-white">
            <h3 className="font-semibold text-blue-900 mb-2">AuditLog</h3>
            <ul className="text-xs space-y-1">
              <li><strong>id</strong>: UUID (PK)</li>
              <li><strong>user_id</strong>: UUID (FK)</li>
              <li><strong>action</strong>: String</li>
              <li><strong>timestamp</strong>: DateTime</li>
              <li><strong>ip_address</strong>: String</li>
              <li><strong>target_type</strong>: String</li>
              <li><strong>target_id</strong>: String</li>
              <li><strong>details</strong>: JSON</li>
            </ul>
          </div>
        </div>
      </div>

      {/* Security Features */}
      <div className={sectionStyle}>
        <h2 className={headerStyle}>
          <Lock size={20} className="inline mr-2" />
          Security Features
        </h2>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="p-3 bg-white border rounded shadow-sm">
            <h3 className="font-semibold text-blue-900 mb-2">Password Security</h3>
            <ul className="text-xs space-y-2">
              <li>• Bcrypt hashing with automatic salt</li>
              <li>• Minimum 12 character length</li>
              <li>• Requirements for uppercase, lowercase, digits, special chars</li>
              <li>• Common password detection</li>
              <li>• Account lockout after failed attempts</li>
            </ul>
          </div>
          
          <div className="p-3 bg-white border rounded shadow-sm">
            <h3 className="font-semibold text-blue-900 mb-2">JWT Implementation</h3>
            <ul className="text-xs space-y-2">
              <li>• HS256 algorithm with secret key</li>
              <li>• Short-lived access tokens (30 min)</li>
              <li>• Refresh token rotation</li>
              <li>• Role and permission claims</li>
              <li>• Expiration verification</li>
            </ul>
          </div>
          
          <div className="p-3 bg-white border rounded shadow-sm">
            <h3 className="font-semibold text-blue-900 mb-2">Audit Trail</h3>
            <ul className="text-xs space-y-2">
              <li>• Comprehensive event logging</li>
              <li>• Login, logout, token refresh events</li>
              <li>• IP address tracking</li>
              <li>• User agent recording</li>
              <li>• Success/failure status</li>
            </ul>
          </div>
          
          <div className="p-3 bg-white border rounded shadow-sm">
            <h3 className="font-semibold text-blue-900 mb-2">Access Control</h3>
            <ul className="text-xs space-y-2">
              <li>• Role-based access control (RBAC)</li>
              <li>• Fine-grained permissions</li>
              <li>• IP-based access restrictions</li>
              <li>• Middleware protection for all API routes</li>
              <li>• Dedicated API key system for services</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AuthFlowDiagram;
