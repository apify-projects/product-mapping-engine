"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.QueueOperationInfo = void 0;
class QueueOperationInfo {
    constructor(requestId, requestOrderNo) {
        Object.defineProperty(this, "requestId", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: requestId
        });
        Object.defineProperty(this, "wasAlreadyPresent", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: void 0
        });
        Object.defineProperty(this, "wasAlreadyHandled", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: void 0
        });
        this.wasAlreadyPresent = requestOrderNo !== undefined;
        this.wasAlreadyHandled = requestOrderNo === null;
    }
}
exports.QueueOperationInfo = QueueOperationInfo;
//# sourceMappingURL=queue_operation_info.js.map