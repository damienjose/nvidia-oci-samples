// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import test from "node:test";
import assert from "node:assert/strict";
import { evaluatePolicy, statusFromChecks } from "../server/policyEngine.ts";
import type { PolicyConfig } from "../server/types.ts";

const policy: PolicyConfig = {
  companyName: "ABC Company",
  currency: "USD",
  requiredFields: ["merchant", "transactionDate", "amount", "currency"],
  receiptRequired: true,
  businessPurposeRequired: true,
  meal: { category: "Meals", attendeesRequiredAbove: 75, warnAbove: 100, tipWarnPercent: 25 },
  lodging: { category: "Lodging", stayDatesWarning: true },
  age: { warnIfOlderThanDays: 90 },
  cash: { warnIfCashPayment: true },
  highAmount: { warnThreshold: 500, blockThreshold: 5000 },
};

test("missing critical fields block downstream staging", () => {
  const checks = evaluatePolicy({ savedFields: { merchant: "Garden Cafe", currency: "USD", receiptFileRef: "/uploads/x.png" } }, "Customer meeting", policy, new Date("2026-05-20T00:00:00Z"));
  assert.equal(statusFromChecks(checks), "blocked");
  assert(checks.some((check) => check.id === "required-amount" && check.decision === "block"));
});

test("meal tip over 25 percent warns but does not block", () => {
  const checks = evaluatePolicy({ savedFields: { merchant: "Harbor Bistro", transactionDate: "2026-05-12", amount: 127.65, currency: "USD", tip: 27.60, category: "Meals", receiptFileRef: "/uploads/x.png" } }, "Customer dinner", policy, new Date("2026-05-20T00:00:00Z"));
  assert.equal(statusFromChecks(checks), "ready_for_review");
  assert(checks.some((check) => check.id === "tip-percent" && check.decision === "warn"));
});

test("lodging without stay dates is warning only", () => {
  const checks = evaluatePolicy({ savedFields: { merchant: "Cloudview Hotel", transactionDate: "2026-05-11", amount: 385.12, currency: "USD", category: "Lodging", receiptFileRef: "/uploads/x.png" } }, "Customer workshop", policy, new Date("2026-05-20T00:00:00Z"));
  assert.equal(statusFromChecks(checks), "ready_for_review");
  assert(checks.some((check) => check.id === "lodging-stay-dates" && check.decision === "warn"));
});
