"""Default-dry V2 entrypoint; exact owner/controller plus a narrow admission guard."""
import json
import check_owned_api as controller
import owned_api_admission_v2 as admission

def preparation(args):
    return admission.prepare(args,controller.HERE,controller.ROOT,controller.need,
        controller.digest,controller.read,controller.pack_metadata)

def main():
    # This is our private entry facade, not a canonical benchmark/ownership patch.
    controller.preparation=preparation
    return controller.main()

if __name__=='__main__':
    try:raise SystemExit(main())
    except (RuntimeError,ValueError) as error:
        print(json.dumps({'refused':True,'error_type':type(error).__name__,'keys_printed':False}))
        raise SystemExit(2)
