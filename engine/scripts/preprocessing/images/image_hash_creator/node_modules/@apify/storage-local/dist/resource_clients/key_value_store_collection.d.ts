export interface KeyValueStoreCollectionClientOptions {
    storageDir: string;
}
export interface KeyValueStoreCollectionData {
    id: string;
    name: string;
    createdAt: Date;
    modifiedAt: Date;
    accessedAt: Date;
}
/**
 * Key-value store collection client.
 */
export declare class KeyValueStoreCollectionClient {
    storageDir: string;
    constructor({ storageDir }: KeyValueStoreCollectionClientOptions);
    list(): Promise<never>;
    getOrCreate(name: string): Promise<KeyValueStoreCollectionData>;
}
//# sourceMappingURL=key_value_store_collection.d.ts.map