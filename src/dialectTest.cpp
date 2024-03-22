#include "mlir/IR/Diagnostics.h"
#include "mlir/Support/LogicalResult.h"

#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"


#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVMPass.h"
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>
#include <string>
#include <system_error>
#include <utility>
#include <mlir/IR/Dialect.h>
#include <mlir/Analysis/FlatLinearValueConstraints.h>
#include <mlir/InitAllDialects.h>
#include <mlir/Dialect/Func/Extensions/AllExtensions.h>

namespace cl = llvm::cl;
using namespace mlir;

static cl::opt<std::string> inputFilename(cl::Positional,
                                          cl::desc("<input toy file>"),
                                          cl::init("-"),
                                          cl::value_desc("filename"));


int loadMLIR(llvm::SourceMgr &sourceMgr, mlir::MLIRContext &context,
             mlir::OwningOpRef<mlir::ModuleOp> &module) {
  // Handle '.toy' input to the compiler.
  assert(llvm::StringRef(inputFilename).ends_with(".mlir"));

  // Otherwise, the input is '.mlir'.
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
      llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
  if (std::error_code ec = fileOrErr.getError()) {
    llvm::errs() << "Could not open input file: " << ec.message() << "\n";
    return -1;
  }

  // Parse the input mlir.
  sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
  module = mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);
  if (!module) {
    llvm::errs() << "Error can't load file " << inputFilename << "\n";
    return 3;
  }
  return 0;
}

int main(int argc, char **argv) {
    mlir::registerAsmPrinterCLOptions();
    mlir::registerMLIRContextCLOptions();
    cl::ParseCommandLineOptions(argc, argv, "parser test\n");
    mlir::DialectRegistry registry;
    mlir::registerAllDialects(registry);
    mlir::func::registerAllExtensions(registry);

    mlir::MLIRContext context(registry);
    mlir::OwningOpRef<mlir::ModuleOp> module;
    llvm::SourceMgr sourceMgr;
    mlir::SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr, &context);
    if (int error = loadMLIR(sourceMgr, context, module))
        return error;

    mlir::PassManager pm(module.get()->getName());

    pm.addNestedPass<func::FuncOp>(createConvertVectorToSCFPass());
    pm.addNestedPass<mlir::func::FuncOp>(mlir::createConvertLinalgToLoopsPass());

    pm.addPass(createLowerAffinePass());
    // Convert SCF to CF (always needed).
    pm.addPass(createConvertSCFToCFPass());
    // Sprinkle some cleanups.
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
    // Convert vector to LLVM (always needed).
    pm.addPass(createConvertVectorToLLVMPass());
    // Convert Math to LLVM (always needed).
    pm.addNestedPass<func::FuncOp>(createConvertMathToLLVMPass());
    // Expand complicated MemRef operations before lowering them.
    pm.addPass(memref::createExpandStridedMetadataPass());
    // The expansion may create affine expressions. Get rid of them.
    pm.addPass(createLowerAffinePass());
    // Convert MemRef to LLVM (always needed).
    pm.addPass(createFinalizeMemRefToLLVMConversionPass());
    // Convert Func to LLVM (always needed).
    pm.addPass(createConvertFuncToLLVMPass());
    // Convert Index to LLVM (always needed).
    pm.addPass(createConvertIndexToLLVMPass());
    // Convert remaining unrealized_casts (always needed).
    pm.addPass(createReconcileUnrealizedCastsPass());

    pm.run(*module);

    module->dump();

    return 0;
}