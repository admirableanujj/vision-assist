CREATE TABLE "users" (
  "id" bigint,
  "guid_user" varchar2,
  "username" varchar2,
  "first_name" varchar2,
  "last_name" varchar2,
  "email" varchar2,
  "role" int,
  "status" varchar2,
  "is_active" bool,
  "created_on" datetime,
  "last_modified_by" varchar2,
  "last_modified_on" datetime
);

CREATE TABLE "user_login" (
  "id" bigint,
  "user_id" bigint,
  "hashed_password" varchar2,
  "created_on" datetime,
  "last_modified_by" varchar2,
  "last_modified_on" datetime
);

CREATE TABLE "user_login_history" (
  "id" bigint,
  "user_id" bigint,
  "login_status" varchar2,
  "created_at" datetime,
  "last_modified_by" varchar2,
  "last_modified_on" datetime
);

CREATE TABLE "roles" (
  "id" bigint,
  "role_name" varchar2,
  "role_desciption" varchar2,
  "created_at" datetime,
  "last_modified_by" varchar2,
  "last_modified_on" datetime
);

CREATE TABLE "permission" (
  "id" bigint,
  "role_id" varchar2,
  "permission_desciption" varchar2,
  "created_on" datetime,
  "last_modified_by" varchar2,
  "last_modified_on" datetime
);

CREATE TABLE "items" (
  "id" varchar,
  "owner_id" bigint,
  "item_name" varchar2,
  "description" varchar2,
  "object_class" varchar2,
  "home_zone_id" bigint,
  "created_on" datetime,
  "last_modified_by" varchar2,
  "last_modified_on" datetime
);

CREATE TABLE "cameras" (
  "id" varchar,
  "owner_id" bigint,
  "name" string,
  "source" string,
  "location" string,
  "created_at" datetime
);

CREATE TABLE "zones" (
  "id" varchar,
  "camera_id" bigint,
  "name" string,
  "x1" integer,
  "y1" integer,
  "x2" integer,
  "y2" integer
);

CREATE TABLE "detections" (
  "id" varchar,
  "camera_id" bigint,
  "object_class" string,
  "confidence" float,
  "zone_name" string,
  "bx1" float,
  "by1" float,
  "bx2" float,
  "by2" float,
  "timestamp" datetime
);

CREATE TABLE "reminders" (
  "id" varchar,
  "owner_id" bigint,
  "text" string,
  "remind_at" datetime,
  "done" boolean,
  "created_at" datetime
);

CREATE TABLE "alerts" (
  "id" varchar,
  "owner_id" bigint,
  "message" string,
  "item_name" string,
  "created_at" datetime,
  "read" boolean
);

CREATE TABLE "query_logs" (
  "id" varchar,
  "owner_id" bigint,
  "item_id" bigint,
  "text" string,
  "intent" string,
  "found" boolean,
  "latency_ms" integer,
  "created_at" datetime
);

CREATE TABLE "item_embeddings" (
  "id" varchar,
  "item_id" bigint,
  "vector" text,
  "model" string,
  "created_at" datetime
);

ALTER TABLE "users" ADD CONSTRAINT "fk_rails_cameras_users" FOREIGN KEY ("id") REFERENCES "cameras" ("owner_id") DEFERRABLE INITIALLY IMMEDIATE;

ALTER TABLE "users" ADD CONSTRAINT "fk_rails_items_users" FOREIGN KEY ("id") REFERENCES "items" ("owner_id") DEFERRABLE INITIALLY IMMEDIATE;

ALTER TABLE "zones" ADD CONSTRAINT "fk_rails_items_zones" FOREIGN KEY ("id") REFERENCES "items" ("home_zone_id") DEFERRABLE INITIALLY IMMEDIATE;

ALTER TABLE "cameras" ADD CONSTRAINT "fk_rails_zones_cameras" FOREIGN KEY ("id") REFERENCES "zones" ("camera_id") DEFERRABLE INITIALLY IMMEDIATE;

ALTER TABLE "cameras" ADD CONSTRAINT "fk_rails_detections_cameras" FOREIGN KEY ("id") REFERENCES "detections" ("camera_id") DEFERRABLE INITIALLY IMMEDIATE;

ALTER TABLE "users" ADD CONSTRAINT "fk_rails_reminders_users" FOREIGN KEY ("id") REFERENCES "reminders" ("owner_id") DEFERRABLE INITIALLY IMMEDIATE;

ALTER TABLE "users" ADD CONSTRAINT "fk_rails_alerts_users" FOREIGN KEY ("id") REFERENCES "alerts" ("owner_id") DEFERRABLE INITIALLY IMMEDIATE;

ALTER TABLE "users" ADD CONSTRAINT "fk_rails_query_logs_users" FOREIGN KEY ("id") REFERENCES "query_logs" ("owner_id") DEFERRABLE INITIALLY IMMEDIATE;

ALTER TABLE "items" ADD CONSTRAINT "fk_rails_query_logs_items" FOREIGN KEY ("id") REFERENCES "query_logs" ("item_id") DEFERRABLE INITIALLY IMMEDIATE;

ALTER TABLE "items" ADD CONSTRAINT "fk_rails_item_embeddings_items" FOREIGN KEY ("id") REFERENCES "item_embeddings" ("item_id") DEFERRABLE INITIALLY IMMEDIATE;

ALTER TABLE "user_login" ADD FOREIGN KEY ("user_id") REFERENCES "users" ("id") DEFERRABLE INITIALLY IMMEDIATE;

ALTER TABLE "user_login_history" ADD FOREIGN KEY ("user_id") REFERENCES "users" ("id") DEFERRABLE INITIALLY IMMEDIATE;
