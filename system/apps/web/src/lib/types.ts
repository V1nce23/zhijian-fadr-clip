export type UserRole = "admin" | "analyst";
// The retired discriminator is retained only for historical reports.
export type MediaType = "image" | "video" | "audio";
export type DetectionDevice = "cpu" | "cuda";
export type ThresholdMode = "balanced" | "low_false_positive" | "low_false_negative" | "custom";
export interface ThresholdModeOption {
  mode: ThresholdMode;
  label: string;
  description: string;
  video_threshold: number;
  frame_threshold: number;
}
export interface ThresholdPolicy {
  version: string;
  mode: ThresholdMode;
  label: string;
  operator: ">";
  threshold: number;
  frame_threshold: number;
  score_unit: "image" | "video_mean";
  video_threshold?: number;
  selection_status: string;
}
export type InputFaceMode = "raw" | "aligned";
export interface DetectionDevices {
  worker_ready: boolean;
  cpu: boolean;
  gpu: boolean;
  gpu_name: string | null;
}
export type AnalysisStatus =
  | "created"
  | "queued"
  | "preprocessing"
  | "running"
  | "aggregating"
  | "completed"
  | "partially_completed"
  | "failed"
  | "cancelled";
export type DetectorStatus =
  | "queued"
  | "running"
  | "succeeded"
  | "failed"
  | "skipped"
  | "timed_out"
  | "cancelled";
export type Verdict =
  | "likely_manipulated"
  | "uncertain"
  | "likely_authentic";

export interface User {
  id: string;
  email: string;
  name: string;
  role: UserRole;
  is_active: boolean;
  created_at: string;
}

export interface Upload {
  id: string;
  original_name: string;
  mime_type: string;
  media_type: MediaType;
  size_bytes: number;
  sha256: string;
  status: "validated" | "deleted";
  created_at: string;
}

export interface DetectorRun {
  id: string;
  detector_id: string;
  model_version: string;
  preprocessing_version: string;
  status: DetectorStatus;
  progress: number;
  score: number | null;
  verdict: Verdict | null;
  evidence: Record<string, unknown> | null;
  error: string | null;
  started_at: string | null;
  completed_at: string | null;
}

export interface Analysis {
  id: string;
  status: AnalysisStatus;
  progress: number;
  requested_device: DetectionDevice | null;
  executed_device: DetectionDevice | null;
  input_face_mode: InputFaceMode | null;
  threshold_policy: ThresholdPolicy | null;
  verdict: Verdict | null;
  score: number | null;
  summary: string | null;
  cancel_requested: boolean;
  started_at: string | null;
  completed_at: string | null;
  created_at: string;
  upload: Upload;
  detector_runs: DetectorRun[];
}

export interface AnalysisListItem {
  id: string;
  status: AnalysisStatus;
  progress: number;
  requested_device: DetectionDevice | null;
  executed_device: DetectionDevice | null;
  input_face_mode: InputFaceMode | null;
  threshold_policy: ThresholdPolicy | null;
  verdict: Verdict | null;
  score: number | null;
  created_at: string;
  completed_at: string | null;
  upload: Upload;
}
