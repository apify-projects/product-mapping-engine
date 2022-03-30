export interface DatasetCollectionClientOptions {
    storageDir: string;
}
export interface DatasetCollectionData {
    id: string;
    name: string;
    createdAt: Date;
    modifiedAt: Date;
    accessedAt: Date;
}
/**
 * Dataset collection client.
 */
export declare class DatasetCollectionClient {
    storageDir: string;
    constructor({ storageDir }: DatasetCollectionClientOptions);
    list(): Promise<never>;
    getOrCreate(name: string): Promise<DatasetCollectionData>;
}
//# sourceMappingURL=dataset_collection.d.ts.map