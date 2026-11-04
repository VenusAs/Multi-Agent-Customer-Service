
# API设计规范

## 一、RESTful 设计原则

### 1.1 URL 命名
- 使用名词复数：/users、/orders
- 使用小写字母和连字符：/user-profiles
- 避免动词：❌ /getUser ✅ /users/{id}
- 层级不超过 3 层：/users/{id}/orders

### 1.2 HTTP 方法
| 方法 | 用途 | 示例 |
|:---|:---|:---|
| GET | 查询 | GET /users |
| POST | 创建 | POST /users |
| PUT | 全量更新 | PUT /users/{id} |
| PATCH | 部分更新 | PATCH /users/{id} |
| DELETE | 删除 | DELETE /users/{id} |

### 1.3 状态码
- 200：成功
- 201：创建成功
- 400：请求参数错误
- 401：未认证
- 403：无权限
- 404：资源不存在
- 500：服务器错误

## 二、请求与响应格式

### 2.1 请求头
Content-Type: application/json
Authorization: Bearer <token>

### 2.2 响应格式
{
  "code": 0,
  "message": "success",
  "data": {...}
}

### 2.3 错误格式
{
  "code": 40001,
  "message": "参数错误：email 格式不正确",
  "data": null
}

## 三、版本控制

- URL 中带版本号：/api/v1/users
- 新版本不兼容旧版本时，升级版本号
- 旧版本至少保留 6 个月

## 四、安全规范

- 所有接口必须 HTTPS
- 敏感数据必须加密传输
- 接口必须限流：单 IP 100 次/分钟
- 必须校验用户权限
