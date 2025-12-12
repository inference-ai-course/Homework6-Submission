/* eslint-env node */

/** @type {import('eslint').Linter.Config} */
module.exports = {
  root: true,
  env: {
    browser: true,
    es2021: true,
    node: true
  },
  parser: "@typescript-eslint/parser",
  parserOptions: {
    ecmaVersion: "latest",
    sourceType: "module",
    ecmaFeatures: { jsx: true }
  },
  plugins: ["@typescript-eslint", "react-hooks", "unicorn"],
  extends: [
    "eslint:recommended",
    "plugin:@typescript-eslint/recommended",
    "plugin:unicorn/recommended"
  ],
  rules: {
    // TypeScript/React ergonomics
    "react/react-in-jsx-scope": "off",
    "react/prop-types": "off",

    // Unicorn tweaks for typical React+TS projects
    "unicorn/no-null": "off",
    "unicorn/prevent-abbreviations": "off",
    "unicorn/filename-case": ["error", {
      cases: { camelCase: true, pascalCase: true }
    }]
  },
  ignorePatterns: [
    "node_modules/",
    "build/",
    "dist/",
    "coverage/",
    "public/",
    "*.config.js",
    "*.config.cjs"
  ]
};
