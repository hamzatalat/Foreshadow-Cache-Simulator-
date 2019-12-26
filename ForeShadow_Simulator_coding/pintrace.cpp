// locations is export PATH=$PATH:/home/hammad/studies/fast/fyp/pin-3.7-97619-g0d0c92f4f-gcc-linux/
// obj-intel64 need to add this to the compile command some how ?
#include "/home/hammad/studies/fast/fyp/pin-3.7-97619-g0d0c92f4f-gcc-linux/source/include/pin/pin.H"
#include <iostream>
#include <fstream>
#include <iomanip>
using namespace std
ofstream TraceFile;
KNOB<string> KnobOutputFile(KNOB_MODE_WRITEONCE, "pintool","o", "pinatrace.out", "specify trace file name");
KNOB<BOOL> KnobValues(KNOB_MODE_WRITEONCE, "pintool","values", "1", "Output memory values reads and written");
static INT32 Usage()
{
    cerr<<"This tool produces a memory address trace.\nFor each (dynamic) instruction reading or writing to memory the the ip and ea are recorded\n";
    cerr<<KNOB_BASE::StringKnobSummary();
    cerr<<endl;
    return -1;
}
static VOID EmitMem(VOID * ea, INT32 size)
{
    if (!KnobValues)
        return;
    switch(size)
    {
      case 0:
        TraceFile<<setw(1);
        break;
      case 1:
        TraceFile<<static_cast<UINT32>(*static_cast<UINT8*>(ea));
        break;
      case 2:
        TraceFile<<*static_cast<UINT16*>(ea);
        break;
      case 4:
        TraceFile<<*static_cast<UINT32*>(ea);
        break;
      case 8:
        TraceFile<<*static_cast<UINT64*>(ea);
        break;
      default:
        TraceFile.unsetf(ios::showbase);
        TraceFile<<setw(1)<<"0x";
        for(INT32 i = 0; i < size; i++)
        {
            TraceFile << static_cast<UINT32>(static_cast<UINT8*>(ea)[i]);
        }
        TraceFile.setf(ios::showbase);
        break;
    }
}
static VOID RecordMem(VOID * ip, CHAR r, VOID * addr, INT32 size, BOOL isPrefetch)
{
    TraceFile<<ip<<": "<<r<<" "<<setw(2+2*sizeof(ADDRINT))<<addr<<" "<<dec<<setw(2)<<size<<" "<<hex<<setw(2+2*sizeof(ADDRINT));
    if (!isPrefetch)
        EmitMem(addr, size);
    TraceFile<<endl;
}
static VOID * WriteAddr;
static INT32 WriteSize;
static VOID RecordWriteAddrSize(VOID * addr, INT32 size)
{
    WriteAddr = addr;
    WriteSize = size;
}
static VOID RecordMemWrite(VOID * ip)
{
    RecordMem(ip, 'W', WriteAddr, WriteSize, false);
}
VOID Instruction(INS ins, VOID *v)
{   
    if (INS_IsMemoryRead(ins) && INS_IsStandardMemop(ins))
    {
        INS_InsertPredicatedCall(
            ins, IPOINT_BEFORE, (AFUNPTR)RecordMem,
            IARG_INST_PTR,
            IARG_UINT32, 'R',
            IARG_MEMORYREAD_EA,
            IARG_MEMORYREAD_SIZE,
            IARG_BOOL, INS_IsPrefetch(ins),
            IARG_END);
    }
    if (INS_HasMemoryRead2(ins) && INS_IsStandardMemop(ins))
    {
        INS_InsertPredicatedCall(
            ins, IPOINT_BEFORE, (AFUNPTR)RecordMem,
            IARG_INST_PTR,
            IARG_UINT32, 'R',
            IARG_MEMORYREAD2_EA,
            IARG_MEMORYREAD_SIZE,
            IARG_BOOL, INS_IsPrefetch(ins),
            IARG_END);
    }
    if (INS_IsMemoryWrite(ins) && INS_IsStandardMemop(ins))
    {
        INS_InsertPredicatedCall(
            ins, IPOINT_BEFORE, (AFUNPTR)RecordWriteAddrSize,
            IARG_MEMORYWRITE_EA,
            IARG_MEMORYWRITE_SIZE,
            IARG_END);
        if (INS_HasFallThrough(ins))
        {
            INS_InsertCall(
                ins, IPOINT_AFTER, (AFUNPTR)RecordMemWrite,
                IARG_INST_PTR,
                IARG_END);
        }
        if (INS_IsBranchOrCall(ins))
        {
            INS_InsertCall(
                ins, IPOINT_TAKEN_BRANCH, (AFUNPTR)RecordMemWrite,
                IARG_INST_PTR,
                IARG_END);
        }
    }
}
VOID Fini(INT32 code, VOID *v)
{
    TraceFile<<"#eof"<<endl;
    TraceFile.close();
}
int main(int argc, char *argv[])
{
    string trace_header=string("#\n# Memory Access Trace Generated By Pin\n#\n");
    if( PIN_Init(argc,argv) )
    {
        return Usage();
    }
    TraceFile.open(KnobOutputFile.Value().c_str());
    TraceFile.write(trace_header.c_str(),trace_header.size());
    TraceFile.setf(ios::showbase);
    INS_AddInstrumentFunction(Instruction, 0);
    PIN_AddFiniFunction(Fini, 0);
    PIN_StartProgram();
    RecordMemWrite(0);
    RecordWriteAddrSize(0, 0);
    return 0;
}
